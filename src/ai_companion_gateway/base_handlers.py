"""gateway 클라이언트 래퍼들의 추상 기반 클래스.

`ai-companion-llm-backend` 의 로컬 추론 핸들러(transformers/gguf/mlx)와
API 프로바이더 클라이언트가 공유하던 `base_handlers` 에서,
프로바이더 클라이언트에 필요한 최소 기반만 여기로 옮겨왔다.
무거운 추론 의존성(torch/transformers/llama.cpp/mlx)은 gateway가 갖지 않는다.
"""

from abc import ABC, abstractmethod
import base64
import random
from io import BytesIO
from typing import Any, BinaryIO, List, Optional, Union

from PIL import Image, ImageFile
from pydantic import SecretStr

from .interfaces.tools import ToolExecutor, ToolSpec
from .tool_calling import DEFAULT_MAX_TOOL_ITERATIONS, run_tool_loop


class UnsupportedTaskError(Exception):
    pass


class BaseModel(ABC):
    def __init__(self, use_langchain: bool = True, image_input: Optional[str | List[str] | Image.Image | List[Image.Image] | ImageFile.ImageFile | List[ImageFile.ImageFile] | Any] = None, audio_input: Optional[str | List[str] | Any] = None, video_input: Optional[str | List[str] | Any] = None, **kwargs):
        self.use_langchain = use_langchain
        self.images: Optional[str | list[str] | Image.Image | list[Image.Image]] = None
        self.audios: Optional[str | list[str]] = None
        self.videos: Optional[str | list[str]] = None
        self.image_input = image_input
        self.audio_input = audio_input
        self.video_input = video_input
        self.enable_streaming = bool(kwargs.get("enable_streaming", False))
        self.use_tools = bool(kwargs.get("use_tools", False))
        # Raw tool schemas (legacy apply_chat_template path). May be a list of
        # OpenAI-style dicts or backend ToolSpec objects.
        self.tools: list = list(kwargs.get("tools", []))

        # Provider-agnostic tool-calling loop settings.
        # ``tool_specs`` are normalized ToolSpec objects; ``tool_executor`` is a
        # callable ``(name, arguments) -> ToolResult | str`` invoked for each
        # requested tool call. ``max_tool_iterations`` caps the loop.
        self.tool_specs: list[ToolSpec] = list(kwargs.get("tool_specs", []))
        self.tool_executor: ToolExecutor | None = kwargs.get("tool_executor", None)
        self.max_tool_iterations = int(kwargs.get("max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS))

        self.max_tokens = int(kwargs.get("max_tokens", 4096))
        self.max_length = int(kwargs.get("max_length", -1))
        self.seed = int(kwargs.get("seed", 42))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.top_k = int(kwargs.get("top_k", 50))
        self.top_p = float(kwargs.get("top_p", 1.0))
        self.repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        self.enable_thinking = bool(kwargs.get("enable_thinking", False))
        self.enable_langchain = False

        self.langchain_integrator = None

        self.use_chunking = bool(kwargs.get("use_chunking", False))
        self.chunk_size = int(kwargs.get("chunk_size", 1024))
        self.extra_kwargs = dict[str, Any](kwargs.get("extra_kwargs", {}))

        self._memory = None

        if self.seed == -1:
            self.seed = random.randint(0, 4294967295)

    @property
    def memory(self):
        """mem0 Memory 인스턴스를 최초 접근 시점에만 초기화한다.

        gateway는 mem0를 하드 의존하지 않으므로, 실제로 메모리를 쓰는
        클라이언트(lmstudio 등)에서만 임포트가 일어난다.
        """
        if self._memory is None:
            from mem0 import Memory

            self._memory = Memory()
        return self._memory

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass


class BaseAPIClientWrapper(BaseModel):
    def __init__(self, selected_model: str, api_key: str | SecretStr | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(use_langchain, image_input, audio_input, **kwargs)
        self.model = selected_model
        self.api_key = api_key

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    # ------------------------------------------------------------------
    # Provider-agnostic tool calling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_specs(tools: list) -> list[ToolSpec]:
        """Coerce a mixed list of ToolSpec/dict/object schemas into ToolSpec."""
        specs: list[ToolSpec] = []
        for tool in tools:
            if isinstance(tool, ToolSpec):
                specs.append(tool)
                continue
            if isinstance(tool, dict):
                # Either an OpenAI function schema or a flat tool dict.
                if tool.get("type") == "function" and "function" in tool:
                    fn = tool["function"]
                    specs.append(
                        ToolSpec(
                            name=fn.get("name", ""),
                            description=fn.get("description", ""),
                            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
                        )
                    )
                else:
                    specs.append(
                        ToolSpec(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}},
                            server=tool.get("server_name") or tool.get("server"),
                        )
                    )
                continue
            # Duck-typed objects (e.g. MCPTool)
            schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters", None) or {"type": "object", "properties": {}}
            if isinstance(schema, list):
                schema = {"type": "object", "properties": {}}
            specs.append(
                ToolSpec(
                    name=getattr(tool, "name", str(tool)),
                    description=getattr(tool, "description", "") or "",
                    parameters=schema,
                    server=getattr(tool, "server_name", None),
                )
            )
        return specs

    def get_tool_specs(self) -> list[ToolSpec]:
        """Resolved, normalized tool specs for this wrapper."""
        if self.tool_specs:
            return self.tool_specs
        return self._normalize_tool_specs(self.tools)

    def can_run_tools(self) -> bool:
        """Whether a tool-calling loop can be executed."""
        return bool(self.get_tool_specs()) and callable(self.tool_executor)

    @staticmethod
    def content_to_openai(content: Any) -> Any:
        """Convert internal multimodal content into OpenAI chat content format."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    url = item.get("url") or item.get("image_url")
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts

    def history_to_openai_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build OpenAI chat messages from the internal history format."""
        messages: list[dict[str, Any]] = []
        for msg in history:
            role = msg.get("role")
            if role not in ("system", "user", "assistant"):
                continue
            messages.append({"role": role, "content": self.content_to_openai(msg.get("content", ""))})
        return messages

    def run_tool_calling(
        self,
        history: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
        system_tool_hint: str | None = None,
        max_tokens: int | None = None,
        **completion_kwargs: Any,
    ) -> str:
        """
        Execute the provider-agnostic tool-calling loop against
        ``self.client.chat.completions`` (OpenAI-compatible endpoint).

        Subclasses that use a non-OpenAI client should override this method.
        """
        if not self.can_run_tools():
            raise RuntimeError("Tool calling is not configured (missing tool_specs or tool_executor)")

        return run_tool_loop(
            client=self.client,
            model=self.model,
            messages=self.history_to_openai_messages(history),
            tool_specs=self.get_tool_specs(),
            executor=self.tool_executor,
            temperature=self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            max_iterations=self.max_tool_iterations,
            extra_body=extra_body,
            system_tool_hint=system_tool_hint,
            **completion_kwargs,
        ).strip()

    def generate_chat_title(self, first_message: str, image_input=None) -> str:
        """
        Generate a chat title based on the first message.
        Uses the API to summarize the message into a short title.
        """
        prompt = f"Create a very short chat title (max 5-7 words) that summarizes the following message. Reply with ONLY the title, no quotes or extra text:\n\n{first_message}"

        history = [{"role": "system", "content": "You are a helpful assistant that creates concise chat titles."}, {"role": "user", "content": prompt}]

        # Temporarily reduce max_tokens for title generation
        original_max_tokens = self.max_tokens
        self.max_tokens = 30

        try:
            title = self.generate_answer(history)
            # Clean up the title
            title = title.strip().strip("\"'").strip()
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception as e:
            from ai_companion_core.logging import logger

            logger.warning(f"Failed to generate chat title via API: {e}")
            # Fallback to truncated first message
            if isinstance(first_message, str):
                return first_message[:50] + "..." if len(first_message) > 50 else first_message
            return "New Chat"
        finally:
            self.max_tokens = original_max_tokens

    @staticmethod
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            if image_path.rsplit(".")[-1] == "jpg" or "jpeg":
                data_mime = "image/jpeg"
            elif image_path.rsplit(".")[-1] == "png":
                data_mime = "image/png"
            elif image_path.rsplit(".")[-1] == "webp":
                data_mime = "image/webp"
            elif image_path.rsplit(".")[-1] == "gif":
                data_mime = "image/gif"

            image = base64.b64encode(image_file.read()).decode("utf-8")

        return image, data_mime


__all__ = ["BaseModel", "BaseAPIClientWrapper", "UnsupportedTaskError"]
