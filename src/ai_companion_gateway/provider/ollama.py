import os
import warnings
from typing import Any, BinaryIO

from typing_extensions import Buffer
from PIL import Image, ImageFile
import ollama
from ollama import Options

SERVER_API_HOST = "127.0.0.1:11434"

from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class OllamaIntegrator(BaseAPIClientWrapper):
    def __init__(self, selected_model: str = None, api_key: str = "not-needed", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.system_prompt = None
        self.user_message = None
        self.chat_history = None

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            self.max_tokens = 4096

        self.server_url = str(kwargs.get("server_url", "http://localhost:11434"))
        # self.client = ollama.Client(host=self.server_url)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="ollama",
                model_name=self.model,
                api_key="not-needed",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
            )
        else:
            self.client = ollama.Client(host=self.server_url)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        else:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

            if self.enable_streaming:
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=Options(
                        seed=self.seed,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repeat_penalty=self.repetition_penalty,
                        num_predict=self.max_tokens,
                    ),
                    stream=True,
                    think="medium" if self.enable_thinking else None,
                )
                in_thinking = False
                answer = ""
                thinking = ""

                for chunk in stream:
                    if chunk.message.thinking:
                        if not in_thinking:
                            in_thinking = True
                            print("Thinking:\n", end="", flush=True)
                        print(chunk.message.thinking, end="", flush=True)
                        thinking += chunk.message.thinking
                    elif chunk.message.content:
                        if in_thinking:
                            in_thinking = False
                            print("\n\nAnswer:\n", end="", flush=True)
                        print(chunk.message.content, end="", flush=True)
                        answer += chunk.message.content

            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=Options(
                        seed=self.seed,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repeat_penalty=self.repetition_penalty,
                        num_predict=self.max_tokens,
                    ),
                    stream=False,
                    think="medium" if self.enable_thinking else None,
                )

                thinking = response.message.thinking if response.message.thinking else ""
                answer = response.message.content

            print(f"\n\n thinking:{thinking}\n answer:{answer}")
            return answer.strip()
