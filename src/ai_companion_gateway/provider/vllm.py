"""
vLLM Provider - Client for connecting to external vLLM OpenAI-compatible API servers.

This module provides a client wrapper for vLLM servers. The vLLM server must be
started separately (e.g., `vllm serve <model> --host 0.0.0.0 --port 8000`).

Usage:
    from ai_companion_gateway.provider.vllm import vLLMClientWrapper

    client = vLLMClientWrapper(
        selected_model="meta-llama/Llama-3.1-8B-Instruct",
        server_url="http://localhost:8000",
        use_langchain=False,
    )

    response = client.generate_answer([
        {"role": "user", "content": "Hello!"}
    ])
"""

from __future__ import annotations

import os
import warnings
from typing import Any, BinaryIO

from typing_extensions import Buffer
from PIL import Image, ImageFile
from openai import OpenAI

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class vLLMClientWrapper(BaseAPIClientWrapper):
    """
    Client wrapper for vLLM OpenAI-compatible API servers.

    vLLM provides an OpenAI-compatible API, so we use the OpenAI client
    to communicate with the server. The server must be started externally.

    Args:
        selected_model: The model name/path being served by vLLM
        api_key: API key (default: "not-needed" for local servers)
        use_langchain: Whether to use LangchainIntegrator for generation
        server_url: vLLM server URL (default: "http://localhost:8000")
        image_input: Optional image input for vision models
        **kwargs: Additional parameters (max_tokens, temperature, top_p, etc.)
    """

    def __init__(self, selected_model: str, api_key: str = "not-needed", use_langchain: bool = False, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.server_url = str(kwargs.get("server_url", "http://localhost:8000"))

        # Normalize server URL (remove trailing slash)
        if self.server_url.endswith("/"):
            self.server_url = self.server_url.rstrip("/")

        # OpenAI client for vLLM's OpenAI-compatible API
        self.client: OpenAI | None = None

        # Handle max_length -> max_tokens conversion
        if self.max_length > 0:
            self.max_tokens = self.max_length

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.load_model()

    def load_model(self):
        """Initialize the OpenAI client or LangchainIntegrator."""
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="vllm-api",
                model_name=self.model,
                api_key=self.api_key,
                base_url=f"{self.server_url}/v1",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
            )
            logger.info(f"vLLM LangchainIntegrator initialized for model: {self.model}")
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=f"{self.server_url}/v1",
            )
            logger.info(f"vLLM OpenAI client initialized: {self.server_url}")

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs) -> str:
        """
        Generate a response based on the conversation history.

        Args:
            history: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        # --- Tool calling path (OpenAI-compatible chat completions) ---
        if self.can_run_tools():
            return self.run_tool_calling(history, extra_body={"repetition_penalty": self.repetition_penalty, "top_k": self.top_k})

        # Prepare messages
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        # Build extra parameters
        extra_body = {}
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.top_k != 50:  # vLLM supports top_k via extra_body
            extra_body["top_k"] = self.top_k

        if self.enable_streaming:
            return self._generate_streaming(messages, extra_body)
        else:
            return self._generate_non_streaming(messages, extra_body)

    def _generate_non_streaming(self, messages: list[dict], extra_body: dict) -> str:
        """Generate response without streaming."""
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
            logger.error(f"vLLM generation error: {e}")
            raise

    def _generate_streaming(self, messages: list[dict], extra_body: dict) -> str:
        """Generate response with streaming."""
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

            print()  # Newline after streaming
            return answer.strip()

        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise

    def check_health(self) -> bool:
        """
        Check if the vLLM server is healthy and responding.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # vLLM provides a /health endpoint
            import requests

            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """
        List available models on the vLLM server.

        Returns:
            List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []


# Legacy aliases for backward compatibility
vLLMIntegrator = vLLMClientWrapper
vLLMAPIIntegrator = vLLMClientWrapper
