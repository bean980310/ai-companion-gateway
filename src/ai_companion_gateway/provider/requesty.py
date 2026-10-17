from __future__ import annotations

import warnings


from huggingface_hub import InferenceClient

from ai_companion_core import logger

from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class RequestyClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(provider="requesty", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, verbose=True)
        else:
            self.client = InferenceClient(base_url="https://router.requesty.ai/v1", api_key=self.api_key)

    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            logger.info(f"[*] Requesty API 요청: {messages}")

            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            answer = chat_completion.choices[0].message.content
            return answer
