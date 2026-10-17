"""Provider-agnostic client factory (gateway entry point).

Instead of a long ``if provider == ...`` chain in the application layer,
every provider registers a *client builder* here.  ``create_client()``
resolves the API key from the environment (if needed) and dispatches to
the matching builder.

The application can also register its own providers (e.g. ``custom:*``
OpenAI-compatible profiles) with :func:`register_client_builder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ai_companion_core import logger
from ai_companion_core.environ_manager import load_env_variables

from .interfaces.providers import PROVIDER_ID
from .provider.anthropic import AnthropicClientWrapper
from .provider.google_genai import GoogleAIClientWrapper
from .provider.huggingface import HuggingfaceInferenceClientWrapper
from .provider.lmstudio import LMStudioIntegrator
from .provider.ollama import OllamaIntegrator
from .provider.openai import OpenAIClientWrapper
from .provider.openrouter import OpenRouterClientWrapper
from .provider.perplexity import PerplexityClientWrapper
from .provider.requesty import RequestyClientWrapper
from .provider.vllm import vLLMClientWrapper
from .provider.xai import XAIClientWrapper
from .registries.provider_registry import ProviderRegistry

# Prefix used for user-defined OpenAI-compatible provider profiles
# (e.g. ``custom:my-vllm-server``).  The application resolves the profile
# and registers a builder per profile with this prefix.
CUSTOM_PREFIX = "custom:"

# A builder receives (model, api_key, **kwargs) and returns a client wrapper.
ClientBuilder = Callable[..., Any]


@dataclass(frozen=True)
class ClientBuilderSpec:
    build: ClientBuilder
    # Environment variable used to resolve the API key when the caller
    # did not pass one explicitly (local providers leave this None).
    api_key_env: Optional[str] = None
    requires_api_key: bool = True
    # Missing key behaviour: raise ValueError if True, otherwise pass api_key=None.
    strict_api_key: bool = True


_BUILDERS: Dict[str, ClientBuilderSpec] = {}


def register_client_builder(
    provider_id: str,
    build: ClientBuilder,
    *,
    api_key_env: Optional[str] = None,
    requires_api_key: bool = True,
    strict_api_key: bool = True,
    overwrite: bool = False,
) -> None:
    """Register a client builder for ``provider_id``.

    ``provider_id`` may be a built-in id (see PROVIDER_ID) or any namespaced
    id such as ``custom:my-profile``.
    """
    if provider_id in _BUILDERS and not overwrite:
        raise ValueError(f"Client builder already registered: {provider_id}")
    _BUILDERS[provider_id] = ClientBuilderSpec(
        build=build,
        api_key_env=api_key_env,
        requires_api_key=requires_api_key,
        strict_api_key=strict_api_key,
    )


def unregister_client_builder(provider_id: str) -> None:
    _BUILDERS.pop(provider_id, None)


def is_supported_provider(provider_id: str) -> bool:
    """Whether the gateway can build a client for ``provider_id``."""
    return provider_id in _BUILDERS


def supported_providers() -> List[str]:
    return list(_BUILDERS.keys())


def resolve_api_key(provider_id: str, api_key: Optional[str] = None) -> Optional[str]:
    """Return ``api_key``, falling back to the provider's environment variable."""
    spec = _BUILDERS.get(provider_id)
    if spec is None or spec.api_key_env is None:
        return api_key
    if api_key:
        return api_key
    return load_env_variables(spec.api_key_env)


def create_client(
    selected_model: str,
    provider: PROVIDER_ID | str,
    *,
    api_key: Optional[str] = None,
    **kwargs: Any,
):
    """Build a provider client wrapper for ``selected_model``.

    Raises:
        ValueError: unknown provider, or a required API key is missing.
    """
    spec = _BUILDERS.get(provider)
    if spec is None:
        raise ValueError(f"지원되지 않는 공급자(gateway에 미등록): {provider}")

    if api_key:
        resolved_key: Optional[str] = api_key
    elif spec.api_key_env:
        resolved_key = load_env_variables(spec.api_key_env)
    else:
        resolved_key = None

    if spec.requires_api_key and not resolved_key and spec.strict_api_key:
        raise ValueError(f"{provider} API Key가 필요합니다. ({spec.api_key_env})")

    logger.info(f"[*] Gateway client: provider={provider}, model={selected_model}")
    return spec.build(selected_model, resolved_key, **kwargs)


# ----------------------------------------------------------------------
# Built-in provider registrations
# ----------------------------------------------------------------------

def _register_builtin_providers() -> None:
    register_client_builder(
        "openai",
        OpenAIClientWrapper,
        api_key_env="OPENAI_API_KEY",
    )
    register_client_builder(
        "anthropic",
        AnthropicClientWrapper,
        api_key_env="ANTHROPIC_API_KEY",
    )
    register_client_builder(
        "google-genai",
        GoogleAIClientWrapper,
        api_key_env="GEMINI_API_KEY",
    )
    register_client_builder(
        "perplexity",
        PerplexityClientWrapper,
        api_key_env="PERPLEXITY_API_KEY",
    )
    register_client_builder(
        "xai",
        XAIClientWrapper,
        api_key_env="XAI_API_KEY",
    )
    register_client_builder(
        "openrouter",
        OpenRouterClientWrapper,
        api_key_env="OPENROUTER_API_KEY",
    )
    register_client_builder(
        "hf-inference",
        HuggingfaceInferenceClientWrapper,
        api_key_env="HF_TOKEN",
    )
    register_client_builder(
        "requesty",
        RequestyClientWrapper,
        api_key_env="REQUESTY_API_KEY",
    )
    # Local providers: no API key required.
    register_client_builder("ollama", OllamaIntegrator, requires_api_key=False)
    register_client_builder("lmstudio", LMStudioIntegrator, requires_api_key=False)
    register_client_builder("vllm", vLLMClientWrapper, requires_api_key=False)


_register_builtin_providers()

__all__ = [
    "CUSTOM_PREFIX",
    "ClientBuilder",
    "ClientBuilderSpec",
    "ProviderRegistry",
    "create_client",
    "is_supported_provider",
    "register_client_builder",
    "resolve_api_key",
    "supported_providers",
    "unregister_client_builder",
]
