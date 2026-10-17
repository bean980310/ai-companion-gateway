from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .logging import logger
from .interfaces.tools import (
    ToolCall,
    ToolExecutor,
    ToolResult,
    ToolSpec,
    build_tool_index,
)

DEFAULT_MAX_TOOL_ITERATIONS = 6

# Providers whose non-langchain path talks to an OpenAI-compatible
# /chat/completions endpoint and therefore supports native tool calling.
# - openai uses the Responses API (handled by run_tool_loop_responses).
# - vllm/openrouter/hf-inference expose chat.completions on an OpenAI-style client.
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "vllm",
    "vllm-api",
    "openrouter",
    "hf-inference",
}

# Providers that expose tools only through the LangchainIntegrator path.
LANGCHAIN_TOOL_PROVIDERS = {
    "anthropic",
    "google-genai",
    "lmstudio",
}


def provider_supports_tools(provider: str) -> bool:
    """Whether native (non-langchain) tool calling is available for a provider."""
    if not provider:
        return False
    provider = provider.lower()
    if provider.startswith("custom:"):
        return True
    return provider in OPENAI_COMPATIBLE_PROVIDERS


def _parse_arguments(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Tool arguments were not valid JSON: {e}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_result(result: Any) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, str):
        return ToolResult(success=True, content=result)
    if isinstance(result, dict) and "success" in result:
        return ToolResult(
            success=bool(result.get("success", True)),
            content=result.get("content"),
            error=result.get("error"),
            content_type=result.get("content_type", "text"),
        )
    return ToolResult(success=True, content=result)


def _execute_tool(executor: ToolExecutor, call: ToolCall) -> ToolResult:
    try:
        return _coerce_result(executor(call.name, call.arguments))
    except Exception as e:  # noqa: BLE001 - surface any tool failure to the model
        logger.error(f"Error executing tool '{call.name}': {e}")
        return ToolResult(success=False, error=str(e))


def run_tool_loop(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    tool_specs: List[ToolSpec],
    executor: ToolExecutor,
    *,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    tool_choice: str = "auto",
    extra_body: Optional[Dict[str, Any]] = None,
    system_tool_hint: Optional[str] = None,
    **completion_kwargs: Any,
) -> str:
    """
    Run an OpenAI-compatible chat-completions tool-calling loop.

    The model is given the tool schemas, any requested calls are executed via
    ``executor``, and the results are fed back until a final text answer is
    produced (or ``max_iterations`` is reached).

    Args:
        client: An OpenAI (or compatible) client exposing ``chat.completions.create``.
        model: Model identifier.
        messages: Conversation messages in OpenAI chat format.
        tool_specs: Tools to expose to the model.
        executor: Callable ``(tool_name, arguments) -> ToolResult | str``.
        temperature: Sampling temperature.
        max_tokens: Max tokens per completion.
        max_iterations: Safety cap on tool-calling rounds.
        tool_choice: OpenAI tool_choice value.
        extra_body: Optional provider-specific extra body (e.g. repetition_penalty).
        system_tool_hint: Optional instruction appended to the system message.
        **completion_kwargs: Additional kwargs forwarded to create().

    Returns:
        The assistant's final text answer, stripped.
    """
    if not tool_specs:
        raise ValueError("run_tool_loop requires at least one ToolSpec")

    tool_index = build_tool_index(tool_specs)
    openai_tools = [spec.to_openai() for spec in tool_specs]

    messages = [dict(m) for m in messages]
    if system_tool_hint:
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": str(messages[0].get("content", "")) + "\n\n" + system_tool_hint,
            }
        else:
            messages.insert(0, {"role": "system", "content": system_tool_hint})

    shared_kwargs: Dict[str, Any] = dict(completion_kwargs)
    if extra_body:
        shared_kwargs.setdefault("extra_body", extra_body)

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            **shared_kwargs,
        )

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            return (message.content or "").strip()

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            spec = tool_index.get(tc.function.name)
            resolved_name = spec.qualified_name if spec else tc.function.name
            call = ToolCall(
                id=tc.id,
                name=resolved_name,
                arguments=_parse_arguments(tc.function.arguments),
                raw_arguments=tc.function.arguments,
            )
            logger.info(f"[tool-loop] call {resolved_name} args={call.arguments}")

            result = _execute_tool(executor, call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.to_text(),
                }
            )

    # Iteration cap reached — request a final answer without tools.
    final = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **shared_kwargs,
    )
    return (final.choices[0].message.content or "").strip()


def run_tool_loop_responses(
    client: Any,
    model: str,
    input_items: List[Dict[str, Any]],
    tool_specs: List[ToolSpec],
    executor: ToolExecutor,
    *,
    instructions: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    tool_choice: str = "auto",
    **create_kwargs: Any,
) -> str:
    """
    OpenAI Responses API variant of the tool-calling loop.

    ``input_items`` should be a list of chat-style messages (role/content) for
    the initial turn; function-call items produced by the model are appended
    automatically between turns.
    """
    if not tool_specs:
        raise ValueError("run_tool_loop_responses requires at least one ToolSpec")

    tool_index = build_tool_index(tool_specs)
    openai_tools = [spec.to_openai() for spec in tool_specs]

    input_items = [dict(item) for item in input_items]

    for _ in range(max_iterations):
        response = client.responses.create(
            model=model,
            input=input_items,
            instructions=instructions,
            tools=openai_tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **create_kwargs,
        )

        output = list(getattr(response, "output", []) or [])
        function_calls = [item for item in output if getattr(item, "type", None) == "function_call"]

        if not function_calls:
            return (getattr(response, "output_text", "") or "").strip()

        # Preserve the model's output items (reasoning, function calls, ...).
        input_items.extend(output)

        for fc in function_calls:
            spec = tool_index.get(fc.name)
            resolved_name = spec.qualified_name if spec else fc.name
            call = ToolCall(
                id=getattr(fc, "call_id", getattr(fc, "id", "")),
                name=resolved_name,
                arguments=_parse_arguments(getattr(fc, "arguments", None)),
                raw_arguments=getattr(fc, "arguments", None),
            )
            logger.info(f"[tool-loop] responses call {resolved_name} args={call.arguments}")
            result = _execute_tool(executor, call)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(fc, "call_id", getattr(fc, "id", "")),
                    "output": result.to_text(),
                }
            )

    final = client.responses.create(
        model=model,
        input=input_items,
        instructions=instructions,
        temperature=temperature,
        max_output_tokens=max_tokens,
        **create_kwargs,
    )
    return (getattr(final, "output_text", "") or "").strip()


def build_tool_specs_from_mcp(tools: List[Any]) -> List[ToolSpec]:
    """
    Convert MCP client tools into backend ToolSpec objects.

    Accepts any objects exposing ``name``, ``description``, ``input_schema``
    and ``server_name`` attributes (e.g. ``ai_companion_agent.mcp.models.MCPTool``).
    """
    specs: List[ToolSpec] = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
        specs.append(
            ToolSpec(
                name=getattr(tool, "name", str(tool)),
                description=getattr(tool, "description", "") or "",
                parameters=schema,
                server=getattr(tool, "server_name", None),
            )
        )
    return specs
