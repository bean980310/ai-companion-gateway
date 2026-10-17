import warnings
from typing import Any

from PIL import Image, ImageFile

from openai import OpenAI

# from openrouter import OpenRouter
from ..base_handlers import BaseAPIClientWrapper

from ai_companion_core import logger

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class OpenRouterClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(provider="openrouter", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, verbose=True)
        else:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        # else: self.client = OpenRouter(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        # --- Tool calling path (OpenAI-compatible chat completions) ---
        if self.can_run_tools():
            return self.run_tool_calling(history, extra_body={"top_k": self.top_k, "repetition_penalty": self.repetition_penalty})

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        logger.info(f"[*] OpenRouter API 요청: {messages}")

        # with self.client as client:
        #     response = client.chat.send(
        #         model=self.model,
        #         messages=messages,

        #     )

        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, extra_body={"top_k": self.top_k, "repetition_penalty": self.repetition_penalty})

        answer = chat_completion.choices[0].message.content
        return answer
