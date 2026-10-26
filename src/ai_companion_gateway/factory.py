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


# ----------------------------------------------------------------------
# OpenAI-compatible endpoints (used by tool-calling loops)
# ----------------------------------------------------------------------
#
# Some consumers (e.g. the MCP tool-calling agent) need a raw OpenAI client
# pointing at a provider's ``/chat/completions`` endpoint rather than a
# wrapper instance.  Endpoint metadata lives here so the application does
# not have to keep its own provider -> (base_url, env) map.

@dataclass(frozen=True)
class OpenAIEndpointSpec:
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    requires_api_key: bool = True
    default_key: Optional[str] = None


_ENDPOINTS: Dict[str, OpenAIEndpointSpec] = {}
_ENDPOINT_ALIASES: Dict[str, str] = {}


def register_openai_endpoint(
    provider_id: str,
    base_url: Optional[str] = None,
    *,
    api_key_env: Optional[str] = None,
    requires_api_key: bool = True,
    default_key: Optional[str] = None,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    """Register OpenAI-compatible endpoint metadata for ``provider_id``."""
    if provider_id in _ENDPOINTS and not overwrite:
        raise ValueError(f"OpenAI endpoint already registered: {provider_id}")
    _ENDPOINTS[provider_id] = OpenAIEndpointSpec(
        base_url=base_url,
        api_key_env=api_key_env,
        requires_api_key=requires_api_key,
        default_key=default_key,
    )
    for alias in aliases:
        _ENDPOINT_ALIASES[alias] = provider_id


def unregister_openai_endpoint(provider_id: str) -> None:
    _ENDPOINTS.pop(provider_id, None)
    for alias, target in list(_ENDPOINT_ALIASES.items()):
        if target == provider_id or alias == provider_id:
            _ENDPOINT_ALIASES.pop(alias, None)


def _resolve_endpoint_spec(provider_id: str) -> Optional[OpenAIEndpointSpec]:
    spec = _ENDPOINTS.get(provider_id)
    if spec is not None:
        return spec
    canonical = _ENDPOINT_ALIASES.get(provider_id)
    if canonical is not None:
        return _ENDPOINTS.get(canonical)
    return None


def supports_openai_tools(provider_id: str) -> bool:
    """Whether ``provider_id`` exposes an OpenAI-compatible endpoint."""
    return _resolve_endpoint_spec(provider_id) is not None


def resolve_openai_endpoint(provider_id: str, api_key: Optional[str] = None) -> tuple[Optional[str], str]:
    """Return ``(base_url, api_key)`` for an OpenAI-compatible provider.

    Raises:
        ValueError: the provider is not registered as OpenAI-compatible, or
            a required API key is missing.
    """
    spec = _resolve_endpoint_spec(provider_id)
    if spec is None:
        raise ValueError(f"Provider '{provider_id}' does not support OpenAI-compatible tool calling")

    key = api_key
    if not key and spec.api_key_env:
        key = load_env_variables(spec.api_key_env)
    if not key:
        key = spec.default_key

    if spec.requires_api_key and not key:
        raise ValueError(f"API key is required for provider '{provider_id}'")

    return spec.base_url, key or "not-needed"


def list_openai_endpoints() -> List[str]:
    return list(_ENDPOINTS.keys())


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

    # OpenAI-compatible endpoints for raw-client consumers (tool loops).
    register_openai_endpoint("openai", None, api_key_env="OPENAI_API_KEY")
    register_openai_endpoint("xai", "https://api.x.ai/v1", api_key_env="XAI_API_KEY")
    register_openai_endpoint("openrouter", "https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY")
    register_openai_endpoint("requesty", "https://router.requesty.ai/v1", api_key_env="REQUESTY_API_KEY")
    register_openai_endpoint("mistralai", "https://api.mistral.ai/v1", api_key_env="MISTRAL_API_KEY")
    register_openai_endpoint(
        "lmstudio",
        "http://localhost:1234/v1",
        api_key_env="LM_API_KEY",
        requires_api_key=False,
        default_key="not-needed",
    )
    register_openai_endpoint(
        "vllm",
        "http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        requires_api_key=False,
        default_key="not-needed",
        aliases=("vllm-api",),
    )


_register_builtin_providers()

__all__ = [
    "CUSTOM_PREFIX",
    "ClientBuilder",
    "ClientBuilderSpec",
    "OpenAIEndpointSpec",
    "ProviderRegistry",
    "create_client",
    "is_supported_provider",
    "list_openai_endpoints",
    "register_client_builder",
    "register_openai_endpoint",
    "resolve_api_key",
    "resolve_openai_endpoint",
    "supported_providers",
    "supports_openai_tools",
    "unregister_client_builder",
    "unregister_openai_endpoint",
]
