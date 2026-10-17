import warnings
from typing import Any

from PIL import Image, ImageFile

from anthropic import Anthropic

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class AnthropicClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.system_prompt: str = None

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="anthropic", model_name=self.model, api_key=self.api_key, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, verbose=True, enable_thinking=self.enable_thinking, image_input=self.image_input
            )
        else:
            self.client = Anthropic(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            # client = anthropic.Client(api_key=self.api_key)

            # Anthropic 메시지 형식으로 변환
            self.system_prompt = next((msg["content"] for msg in history[:1] if msg["role"] == "system"), None)

            # if self.system_prompt is not None:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[1:-1]]
            # else:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]

            # messages = [{"role": msg['role'], "content": msg['content']} for msg in history[(1 if self.system_prompt else 0):-1]]
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[bool(self.system_prompt) : -1]]

            # messages = []
            # for msg in history[:-1]:
            #     if msg["role"] == "system":
            #         self.system = msg["content"]
            #         # continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
            #     else:
            #         messages.append({
            #         "role": msg["role"],
            #         "content": msg["content"]
            #     })

            if self.image_input is not None:
                _, mime_type = self.encode_image(self.image_input)
                with open(self.image_input, "rb") as f:
                    image = self.client.beta.files.upload(file=(self.image_input.split("/")[-1], f, mime_type))
                new_message = {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "file", "file_id": image.id}}, {"type": "text", "text": history[-1]["content"]["text"]}],
                }
                messages.append(new_message)
            else:
                messages.append({"role": "user", "content": history[-1]["content"]})

            logger.info(f"[*] Anthropic API 요청: {messages}")

            if self.enable_streaming is True:
                answer = ""
                with self.client.beta.messages.stream(
                    model=self.model,
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    # frequency_penalty=repetition_penalty,
                    max_tokens=self.max_tokens,
                    thinking={"type": "enabled", "budget_tokens": 10000} if self.enable_thinking else {"type": "disabled"},
                    betas=["files-api-2025-04-14", "skills-2025-10-02", "code_execution_20250825", "web_search_20260209", "mcp-client-2025-11-20"],
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        answer += text

            else:
                response = self.client.beta.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    # frequency_penalty=repetition_penalty,
                    max_tokens=self.max_tokens,
                    thinking={"type": "enabled", "budget_tokens": 10000} if self.enable_thinking else {"type": "disabled"},
                    betas=["files-api-2025-04-14", "skills-2025-10-02", "code_execution_20250825", "web_search_20260209", "mcp-client-2025-11-20"],
                )
                answer = response.content[0].text

            return answer.strip()
