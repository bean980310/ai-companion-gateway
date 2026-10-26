"""Generic client for OpenAI-compatible API servers.

Used by local/self-hosted providers such as vLLM, omlx and SGLang, which all
expose ``/v1/chat/completions`` and ``/v1/models``.  The server is expected to
run separately; this wrapper only talks HTTP.

Usage::

    from ai_companion_gateway.provider.openai_compatible import (
        OpenAICompatibleClientWrapper,
    )

    client = OpenAICompatibleClientWrapper(
        "my-model",
        base_url="http://localhost:8001/v1",
    )
"""

from __future__ import annotations

import os
from typing import Any, BinaryIO

from typing_extensions import Buffer
from PIL import Image, ImageFile
from openai import OpenAI

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper


class OpenAICompatibleClientWrapper(BaseAPIClientWrapper):
    """Client wrapper for any OpenAI-compatible chat-completions server.

    Args:
        selected_model: Model name served by the endpoint.
        api_key: API key (defaults to ``"not-needed"`` for local servers).
        use_langchain: Whether to use LangchainIntegrator for generation.
        base_url: Full base URL, e.g. ``http://localhost:8001/v1``.  ``/v1`` is
            appended when missing.
        server_url: Alternative to ``base_url`` (``/v1`` is appended).
        **kwargs: Additional generation parameters (max_tokens, temperature, ...).
    """

    default_base_url: str = "http://localhost:8000/v1"
    langchain_provider: str = "vllm-api"

    def __init__(
        self,
        selected_model: str,
        api_key: str = "not-needed",
        use_langchain: bool = False,
        image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None,
        base_url: str | None = None,
        server_url: str | None = None,
        **kwargs,
    ):
        super().__init__(selected_model, api_key or "not-needed", use_langchain, image_input, **kwargs)

        resolved = str(base_url or server_url or self.default_base_url).rstrip("/")
        if not resolved.endswith("/v1"):
            resolved = f"{resolved}/v1"
        self.base_url = resolved
        self.server_url = resolved[: -len("/v1")]

        self.client: OpenAI | None = None

        if self.max_length > 0:
            self.max_tokens = self.max_length

        if self.use_langchain:
            self.enable_langchain = True

        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            try:
                from langchain_integrator import LangchainIntegrator
            except ImportError:
                logger.warning("langchain_integrator is not installed; falling back to the OpenAI client.")
                self.enable_langchain = False
            else:
                self.langchain_integrator = LangchainIntegrator(
                    provider=self.langchain_provider,
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    verbose=True,
                )
                logger.info(f"LangchainIntegrator initialized for model: {self.model} ({self.base_url})")
                return

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"OpenAI-compatible client initialized: {self.base_url}")

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs) -> str:
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        # --- Tool calling path (OpenAI-compatible chat completions) ---
        if self.can_run_tools():
            return self.run_tool_calling(history, extra_body={"repetition_penalty": self.repetition_penalty, "top_k": self.top_k})

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        extra_body = {}
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.top_k != 50:
            extra_body["top_k"] = self.top_k

        if self.enable_streaming:
            return self._generate_streaming(messages, extra_body)
        return self._generate_non_streaming(messages, extra_body)

    def _generate_non_streaming(self, messages: list[dict], extra_body: dict) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed if self.seed != -1 else None,
                extra_body=extra_body if extra_body else None,
            )
            answer = response.choices[0].message.content
            return answer.strip() if answer else ""
        except Exception as e:
            logger.error(f"OpenAI-compatible generation error ({self.base_url}): {e}")
            raise

    def _generate_streaming(self, messages: list[dict], extra_body: dict) -> str:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed if self.seed != -1 else None,
                extra_body=extra_body if extra_body else None,
                stream=True,
            )

            answer = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    answer += content

            print()
            return answer.strip()
        except Exception as e:
            logger.error(f"OpenAI-compatible streaming error ({self.base_url}): {e}")
            raise

    def check_health(self) -> bool:
        """Check whether the server responds on ``/health``."""
        try:
            import requests

            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed ({self.base_url}): {e}")
            return False

    def list_models(self) -> list[str]:
        """List models exposed by the server."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models ({self.base_url}): {e}")
            return []


__all__ = ["OpenAICompatibleClientWrapper"]
