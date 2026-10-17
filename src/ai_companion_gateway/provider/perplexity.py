import warnings
from typing import Any, Optional

from PIL import Image, ImageFile

from perplexity import Perplexity
from perplexity._streaming import Stream
from perplexity.types.stream_chunk import StreamChunk

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class PerplexityClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="perplexity", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, verbose=True, enable_thinking=self.enable_thinking, image_input=self.image_input
            )
        else:
            self.client = Perplexity(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[:-1]]

            if self.image_input is not None:
                image, mime_type = self.encode_image(self.image_input)
                new_message = {
                    "role": "user",
                    "content": [{"type": "text", "text": history[-1]["content"]["text"]}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image}"}}],
                }
                messages.append(new_message)
            else:
                messages.append({"role": "user", "content": history[-1]["content"]})

            logger.info(f"[*] Perplexity API 요청: {messages}")

            if self.enable_streaming:
                stream = self.client.responses.create(
                    input=messages,
                    model=self.model,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_output_tokens=self.max_tokens,
                    frequency_penalty=self.repetition_penalty,
                    presence_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    reasoning_effort="medium" if self.enable_thinking else None,
                    stream=True,
                )
                answer = ""
                search_results = []
                usage_info = None
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content_piece = chunk.choices[0].delta.content
                        answer += content_piece
                        print(content_piece, end="", flush=True)

                    if hasattr(chunk, "search_results") and chunk.search_results:
                        search_results = chunk.search_results

                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_info = chunk.usage

                    if chunk.choices[0].finish_reason:
                        print(f"\n\nSearch Results: {search_results}")
                        print(f"Usage: {usage_info}")

                print(f"\n\nSearch Results: {search_results}")
                print(f"Usage: {usage_info}")
                print(answer)

            else:
                completion = self.client.chat.completions.create(
                    messages=messages, model=self.model, top_p=self.top_p, top_k=self.top_k, max_tokens=self.max_tokens, frequency_penalty=self.repetition_penalty, presence_penalty=self.repetition_penalty, temperature=self.temperature, reasoning_effort="medium" if self.enable_thinking else None, stream=False
                )
                answer = completion.choices[0].message.content

            return answer.strip()
