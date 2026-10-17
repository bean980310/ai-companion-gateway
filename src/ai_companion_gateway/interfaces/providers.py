from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, List, Literal

PROVIDER_ID = Literal["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "vllm", "self-provided"]

@dataclass(frozen=True)
class ProviderCapabilities:
    chat: bool = True
    embeddings: bool = False
    vision: bool = False
    tools: bool = False
    # Whether tools are driven by the backend's OpenAI-compatible tool loop
    # (as opposed to only being exposed through LangchainIntegrator).
    tools_native: bool = False
    json_mode: bool = False
    streaming: bool = True

@dataclass(frozen=True)
class ProviderSpec:
    id: PROVIDER_ID
    display_name: str
    requires_api_key: bool
    base_url_hint: Optional[str] = None
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)

class BaseProviderAdapter(Protocol):
    spec: ProviderSpec
    def list_models(self) -> List[str]:
        ...