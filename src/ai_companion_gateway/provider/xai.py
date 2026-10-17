from __future__ import annotations

import warnings
from typing import Any

from PIL import Image, ImageFile

try:
    import xai_sdk
except ImportError:
    warnings.warn("xai_sdk is not Installed")
except Exception as e:
    warnings.warn(f"xai_sdk error: {e}")


from ai_companion_core import logger

from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class XAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(provider="xai", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, verbose=True)
        else:
            self.client = xai_sdk.Client(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            logger.info(f"[*] XAI API 요청: {messages}")

            response = self.client.chat.create(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, messages=messages, top_p=self.top_p, top_logprobs=self.top_k, frequency_penalty=self.repetition_penalty, presence_penalty=self.repetition_penalty)

            if self.enable_streaming:
                answer = ""
                for stream, chunk in response.stream():
                    answer += chunk.content

            else:
                answer = response.sample().content

            return answer
