from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolSpec:
    """
    A callable tool exposed to the model.

    The schema is provider-agnostic: it mirrors the OpenAI function-tool shape
    so that it can be reused by any OpenAI-compatible endpoint.
    """

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    server: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.server}__{self.name}" if self.server else self.name

    def to_openai(self) -> Dict[str, Any]:
        schema = self.parameters or {"type": "object", "properties": {}}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": sanitize_tool_name(self.qualified_name),
                "description": self.description or self.name,
                "parameters": schema,
            },
        }


@dataclass
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: Optional[str] = None


@dataclass
class ToolResult:
    """The outcome of executing a tool."""

    success: bool = True
    content: Any = None
    error: Optional[str] = None
    content_type: str = "text"  # text, image, json, ...

    def to_text(self) -> str:
        if not self.success:
            return f"Error: {self.error or 'tool call failed'}"
        if self.content is None:
            return "(no content)"
        if self.content_type in ("text", "json"):
            return str(self.content)
        if self.content_type == "image":
            return "[image content received]"
        return str(self.content)


# An executor receives (tool_name, arguments) and returns a ToolResult, a
# plain string, or any other value that can be stringified.
ToolExecutor = Callable[[str, Dict[str, Any]], Union[ToolResult, str, Any]]

_FUNCTION_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """Sanitize a tool name into a valid OpenAI function name (max 64 chars)."""
    return _FUNCTION_NAME_RE.sub("_", name)[:64]


def build_tool_index(tool_specs: List[ToolSpec]) -> Dict[str, ToolSpec]:
    """Map sanitized, wire-safe function names back to their ToolSpec."""
    index: Dict[str, ToolSpec] = {}
    for spec in tool_specs:
        fn_name = sanitize_tool_name(spec.qualified_name)
        if fn_name in index:
            suffix = 1
            base = fn_name
            while f"{base}_{suffix}" in index:
                suffix += 1
            fn_name = f"{base}_{suffix}"
        index[fn_name] = spec
    return index
