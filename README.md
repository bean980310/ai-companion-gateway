# ai-companion-gateway

Provider-agnostic LLM gateway for AI Companion, extracted from
`ai-companion-llm-backend`.

## What lives here

- `provider/` — API client wrappers (`OpenAIClientWrapper`, `AnthropicClientWrapper`,
  `OllamaIntegrator`, `LMStudioIntegrator`, `vLLMClientWrapper`, ...) built on
  `base_handlers.BaseAPIClientWrapper`.
- `factory.py` — `create_client(model, provider)` gateway entry point plus
  `register_client_builder()` so applications can register their own providers
  (e.g. `custom:*` OpenAI-compatible profiles) instead of if/elif chains.
  `register_openai_endpoint()` / `resolve_openai_endpoint()` expose the
  `(base_url, api_key)` metadata needed by raw OpenAI-client consumers such as
  tool-calling loops.
- `interfaces/providers.py`, `registries/provider_registry.py` — provider specs
  and capability registry.
- `interfaces/tools.py`, `tool_calling.py` — provider-agnostic tool-calling loop.

`ai-companion-llm-backend` depends on this package and keeps only the local
inference handlers (transformers / GGUF / MLX / vLLM-embedded).

## Usage

```python
from ai_companion_gateway import create_client, ProviderRegistry

client = create_client("gpt-4o", "openai")  # resolves OPENAI_API_KEY from ~/.ai-companion/.env
answer = client.generate_answer(history)
```
