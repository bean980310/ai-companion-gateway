"""AI Companion LLM Gateway.

Provider-agnostic access layer for API-based LLM providers (OpenAI,
Anthropic, Google, Ollama, LM Studio, vLLM, ...).  Extracted from
``ai-companion-llm-backend``; the backend keeps only the local inference
handlers and depends on this package for provider clients.

Typical use::

    from ai_companion_gateway import create_client

    client = create_client("gpt-4o", "openai")
    answer = client.generate_answer(history)
"""

from .base_handlers import BaseAPIClientWrapper, BaseModel, UnsupportedTaskError
from .interfaces.providers import PROVIDER_ID, ProviderCapabilities, ProviderSpec
from .interfaces.tools import ToolCall, ToolExecutor, ToolResult, ToolSpec
from .registries.provider_registry import ProviderRegistry
from .tool_calling import (
    build_tool_specs_from_mcp,
    provider_supports_tools,
    run_tool_loop,
    run_tool_loop_responses,
)
from .provider import (
    AnthropicClientWrapper,
    GoogleAIClientWrapper,
    HuggingfaceInferenceClientWrapper,
    LMStudioIntegrator,
    OllamaIntegrator,
    OpenAIClientWrapper,
    OpenRouterClientWrapper,
    OpenAICompatibleClientWrapper,
    RequestyClientWrapper,
    XAIClientWrapper,
    vLLMClientWrapper,
)
from .factory import (
    CUSTOM_PREFIX,
    ClientBuilder,
    ClientBuilderSpec,
    OpenAIEndpointSpec,
    create_client,
    is_supported_provider,
    list_openai_endpoints,
    register_client_builder,
    register_openai_endpoint,
    resolve_api_key,
    resolve_openai_endpoint,
    supported_providers,
    supports_openai_tools,
    unregister_client_builder,
    unregister_openai_endpoint,
)
from .listing import (
    ModelLister,
    ModelListingError,
    list_all_models,
    list_model_providers,
    list_models,
    register_model_lister,
    supports_model_listing,
    unregister_model_lister,
)

__all__ = [
    # Base classes
    "BaseModel",
    "BaseAPIClientWrapper",
    "UnsupportedTaskError",
    # Provider clients
    "AnthropicClientWrapper",
    "GoogleAIClientWrapper",
    "HuggingfaceInferenceClientWrapper",
    "LMStudioIntegrator",
    "OllamaIntegrator",
    "OpenAIClientWrapper",
    "OpenRouterClientWrapper",
    "OpenAICompatibleClientWrapper",
    "RequestyClientWrapper",
    "XAIClientWrapper",
    "vLLMClientWrapper",
    # Factory / registry
    "CUSTOM_PREFIX",
    "ClientBuilder",
    "ClientBuilderSpec",
    "OpenAIEndpointSpec",
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
    # Model listing
    "ModelLister",
    "ModelListingError",
    "list_all_models",
    "list_model_providers",
    "list_models",
    "register_model_lister",
    "supports_model_listing",
    "unregister_model_lister",
    "ProviderRegistry",
    "ProviderSpec",
    "ProviderCapabilities",
    "PROVIDER_ID",
    # Tools / tool-calling
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "run_tool_loop",
    "run_tool_loop_responses",
    "build_tool_specs_from_mcp",
    "provider_supports_tools",
]

__version__ = "0.3.0"
