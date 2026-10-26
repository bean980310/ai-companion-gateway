"""
vLLM Provider - Client for connecting to external vLLM OpenAI-compatible API servers.

This module provides a client wrapper for vLLM servers. The vLLM server must be
started separately (e.g., `vllm serve <model> --host 0.0.0.0 --port 8000`).

The implementation lives in :mod:`ai_companion_gateway.provider.openai_compatible`;
this class only pins the default vLLM endpoint.

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

from .openai_compatible import OpenAICompatibleClientWrapper


class vLLMClientWrapper(OpenAICompatibleClientWrapper):
    """vLLM OpenAI-compatible server client."""

    default_base_url: str = "http://localhost:8000/v1"
    langchain_provider: str = "vllm-api"


# Legacy aliases for backward compatibility
vLLMIntegrator = vLLMClientWrapper
vLLMAPIIntegrator = vLLMClientWrapper

__all__ = ["vLLMClientWrapper", "vLLMIntegrator", "vLLMAPIIntegrator"]
