import warnings
from typing import Any

from PIL import Image, ImageFile

from google import genai
from google.genai import types

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class GoogleAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.system_prompt: str = None

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="google_genai", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, verbose=True, enable_thinking=self.enable_thinking, image_input=self.image_input
            )
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            self.system_prompt = next((msg["content"] for msg in history[:1] if msg["role"] == "system"), None)
            # if self.system_prompt is not None:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[1:-1]]
            # else:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]

            # messages = [{"role": msg['role'], "content": msg['content']} for msg in history[(1 if self.system_prompt else 0):-1]]
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[bool(self.system_prompt) : -1]]

            if self.image_input is not None:
                image = self.client.files.upload(file=self.image_input)
                new_message = {
                    "role": "user",
                    "content": [{"type": "text", "text": history[-1]["content"]["text"]}, {"type": "file_data", "file_data": {"mime_type": image.mime_type, "file_uri": image.uri}}],
                }
                messages.append(new_message)
            else:
                messages.append({"role": "user", "content": history[-1]["content"]})

            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p if self.enable_thinking is False else None,
                top_k=self.top_k,
                frequency_penalty=self.repetition_penalty if self.enable_thinking is False else None,
                presence_penalty=self.repetition_penalty if self.enable_thinking is False else None,
                thinking_config=types.ThinkingConfig(thinking_level="high" if self.enable_thinking else "low", thinking_budget=-1 if self.enable_thinking else 0),
            )
            logger.info(f"[*] Google API 요청: {messages}")
            if self.enable_streaming:
                stream = self.client.models.generate_content_stream(model=self.model, contents=messages, config=config)
                answer = ""
                for x in stream:
                    print(x.text, end="", flush=True)
                    answer += x.text
            else:
                response = self.client.models.generate_content(model=self.model, contents=messages, config=config)
                answer = response.text

            return answer.strip()
