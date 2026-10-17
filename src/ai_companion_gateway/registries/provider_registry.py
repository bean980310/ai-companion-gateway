from __future__ import annotations
from typing import Dict, List
from ..interfaces.providers import ProviderSpec, ProviderCapabilities, PROVIDER_ID as ProviderId


class ProviderRegistry:
    _providers: Dict[ProviderId, ProviderSpec] = {}

    @classmethod
    def register(cls, spec: ProviderSpec):
        if spec.id in cls._providers:
            raise ValueError(f"Provider already registered: {spec.id}")
        cls._providers[spec.id] = spec

    @classmethod
    def get(cls, provider_id: ProviderId) -> ProviderSpec:
        try:
            return cls._providers[provider_id]
        except KeyError:
            raise KeyError(f"Unknown provider: {provider_id}")

    @classmethod
    def list(cls) -> List[ProviderSpec]:
        return list(cls._providers.values())

    @classmethod
    def list_ids(cls) -> List[ProviderId]:
        return list(cls._providers.keys())

    @classmethod
    def get_capabilities(cls, provider_id: ProviderId) -> ProviderCapabilities:
        """Capabilities for a provider; permissive default if unregistered."""
        spec = cls._providers.get(provider_id)
        if spec is None:
            return ProviderCapabilities()
        return spec.capabilities

    @classmethod
    def supports_tools(cls, provider_id: ProviderId) -> bool:
        """Whether the provider advertises tool-calling support."""
        return cls.get_capabilities(provider_id).tools

    @classmethod
    def register_defaults(cls, *, overwrite: bool = False):
        """Return the built-in provider specs (id -> ProviderSpec)."""
        return {spec.id: spec for spec in cls._providers.values()}


# ---- 실제 등록(예시) ----
ProviderRegistry.register(
    ProviderSpec(
        id="openai",
        display_name="OpenAI",
        requires_api_key=True,
        base_url_hint="https://api.openai.com/v1",
        default_kwargs={"temperature": 1.0, "top_p": 1.0},
        capabilities=ProviderCapabilities(
            chat=True, embeddings=True, vision=True, tools=True, tools_native=True, json_mode=True, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="anthropic",
        display_name="Anthropic",
        requires_api_key=True,
        base_url_hint="https://api.anthropic.com/v1",
        default_kwargs={"temperature": 1.0},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, tools_native=False, json_mode=False, embeddings=False, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="google-genai",
        display_name="Google AI",
        requires_api_key=True,
        base_url_hint="https://generativelanguage.googleapis.com/v1beta",
        default_kwargs={"temperature": 1.0, "top_p": 1, "top_k": 20},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, tools_native=False, json_mode=False, embeddings=False, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="ollama",
        display_name="Ollama (Local)",
        requires_api_key=False,
        base_url_hint="http://localhost:11434",
        default_kwargs={"temperature": 0.7, "top_p": 1, "top_k": 20, "repeat_penalty": 1.05},
        capabilities=ProviderCapabilities(
            chat=True, vision=False, tools=False, embeddings=True, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="vllm",
        display_name="vLLM (Local)",
        requires_api_key=False,
        base_url_hint="http://localhost:8000",
        default_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.0},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, tools_native=True, embeddings=False, json_mode=True, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="lmstudio",
        display_name="LM Studio (Local)",
        requires_api_key=False,
        base_url_hint="http://localhost:1234/v1",
        default_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 50},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, tools_native=False, embeddings=False, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="openrouter",
        display_name="OpenRouter",
        requires_api_key=True,
        base_url_hint="https://openrouter.ai/api/v1",
        default_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 50},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, tools_native=True, embeddings=False, streaming=True
        ),
    )
)