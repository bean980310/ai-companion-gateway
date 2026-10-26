"""Model listing for gateway providers.

Used by the application's "All providers" model picker: instead of the
application knowing how to query every provider, it asks the gateway for a
merged catalog.  Listers are registered per provider and can be extended by
the application (e.g. for ``custom:*`` OpenAI-compatible profiles).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from ai_companion_core import logger

from .factory import (
    resolve_api_key,
    resolve_openai_endpoint,
    supports_openai_tools,
)

# (api_key, base_url) -> model ids
ModelLister = Callable[[Optional[str], Optional[str]], List[str]]

_LISTERS: Dict[str, ModelLister] = {}
_LISTER_ALIASES: Dict[str, str] = {}

_DEFAULT_TIMEOUT = 15.0

# Chat-capable OpenAI models (mirrors the application's provider list filter).
_OPENAI_CHAT_PATTERN = ("gpt-4o", "gpt-4.1", "gpt-5", "gpt-oss")
_OPENAI_EXCLUDE = (
    "image",
    "realtime",
    "tts",
    "audio",
    "transcribe",
    "codex",
    "search",
    "preview",
)
_OPENAI_EXCLUDE_MODELS = (
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
)


class ModelListingError(RuntimeError):
    """Raised when a provider cannot list models."""


def register_model_lister(
    provider_id: str,
    lister: ModelLister,
    *,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    """Register a model lister for ``provider_id``."""
    if provider_id in _LISTERS and not overwrite:
        raise ValueError(f"Model lister already registered: {provider_id}")
    _LISTERS[provider_id] = lister
    for alias in aliases:
        _LISTER_ALIASES[alias] = provider_id


def unregister_model_lister(provider_id: str) -> None:
    _LISTERS.pop(provider_id, None)
    for alias, target in list(_LISTER_ALIASES.items()):
        if target == provider_id or alias == provider_id:
            _LISTER_ALIASES.pop(alias, None)


def _resolve_lister(provider_id: str) -> Optional[ModelLister]:
    lister = _LISTERS.get(provider_id)
    if lister is not None:
        return lister
    canonical = _LISTER_ALIASES.get(provider_id)
    if canonical is not None:
        return _LISTERS.get(canonical)
    return None


def supports_model_listing(provider_id: str) -> bool:
    return _resolve_lister(provider_id) is not None


def list_model_providers() -> List[str]:
    """Providers that can be queried for a model list (registration order)."""
    return list(_LISTERS.keys())


def list_models(provider_id: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
    """Return the model ids exposed by ``provider_id``.

    Raises:
        ModelListingError: unknown provider or the provider call failed.
    """
    lister = _resolve_lister(provider_id)
    if lister is None:
        raise ModelListingError(f"Provider '{provider_id}' does not support model listing")

    key = api_key
    resolved_base_url = base_url

    if supports_openai_tools(provider_id):
        try:
            endpoint_base_url, endpoint_key = resolve_openai_endpoint(provider_id, api_key)
        except ValueError as e:
            raise ModelListingError(str(e)) from e
        resolved_base_url = resolved_base_url or endpoint_base_url
        key = key or endpoint_key
    elif key is None:
        key = resolve_api_key(provider_id, api_key)

    try:
        return lister(key, resolved_base_url)
    except Exception as e:  # noqa: BLE001 - surface as a domain error
        raise ModelListingError(f"{provider_id}: {e}") from e


def list_all_models(
    providers: Optional[List[str]] = None,
    *,
    api_key: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, List[str]]:
    """Query every registered provider in parallel.

    Failed providers are logged and omitted; the result preserves the
    provider registration order.
    """
    provider_ids = list(providers) if providers is not None else list_model_providers()
    if not provider_ids:
        return {}

    results: Dict[str, List[str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(provider_ids)))) as pool:
        futures = {pool.submit(list_models, pid, api_key=api_key): pid for pid in provider_ids}
        for future, provider_id in futures.items():
            try:
                models = future.result()
            except ModelListingError as e:
                logger.warning(f"모델 목록 조회 실패 ({provider_id}): {e}")
                continue
            results[provider_id] = models

    ordered = {pid: results[pid] for pid in provider_ids if pid in results}
    return ordered


# ----------------------------------------------------------------------
# Built-in listers
# ----------------------------------------------------------------------

def _list_openai_compatible(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=_DEFAULT_TIMEOUT)
    return [m.id for m in client.models.list().data]


def _list_openai_chat(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    models: List[str] = []
    for model_id in _list_openai_compatible(api_key, base_url):
        lowered = model_id.lower()
        include = any(k in lowered for k in _OPENAI_CHAT_PATTERN)
        exclude_type = all(k not in lowered for k in _OPENAI_EXCLUDE)
        exclude_model = all(k not in lowered for k in _OPENAI_EXCLUDE_MODELS)
        if include and exclude_type and exclude_model:
            models.append(model_id)
    return models


def _list_anthropic(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    return [m.id for m in client.beta.models.list().data]


def _list_google_genai(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    models: List[str] = []
    for m in client.models.list().page:
        name = m.name
        lowered = name.lower()
        include = any(k in lowered for k in ("gemini", "gemma"))
        exclude_type = all(k not in lowered for k in ("embedding", "tts", "exp"))
        exclude_model = "gemini-2.0" not in lowered
        if include and exclude_type and exclude_model and "generateContent" in m.supported_actions:
            models.append(name)
    return models


def _list_ollama(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    import ollama

    client = ollama.Client(host=base_url or "http://localhost:11434")
    return [m.model for m in client.list().models]


def _list_perplexity(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    import requests

    response = requests.get("https://api.perplexity.ai/v1/models", timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]


def _list_mistralai(api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    from mistralai.client import Mistral

    client = Mistral(api_key=api_key)
    models: List[str] = []
    for m in client.models.list().data:
        capabilities = getattr(m, "capabilities", None)
        if capabilities is not None and not getattr(capabilities, "completion_chat", False):
            continue
        if getattr(m, "deprecation", None):
            continue
        models.append(m.id)
    return models


def _register_builtin_listers() -> None:
    register_model_lister("openai", _list_openai_chat)
    register_model_lister("xai", _list_openai_compatible)
    register_model_lister("openrouter", _list_openai_compatible)
    register_model_lister("requesty", _list_openai_compatible)
    register_model_lister("lmstudio", _list_openai_compatible)
    register_model_lister("vllm", _list_openai_compatible, aliases=("vllm-api",))
    register_model_lister("anthropic", _list_anthropic)
    register_model_lister("google-genai", _list_google_genai)
    register_model_lister("ollama", _list_ollama)
    register_model_lister("perplexity", _list_perplexity)
    register_model_lister("mistralai", _list_mistralai)


_register_builtin_listers()

__all__ = [
    "ModelLister",
    "ModelListingError",
    "list_all_models",
    "list_model_providers",
    "list_models",
    "register_model_lister",
    "supports_model_listing",
    "unregister_model_lister",
]
