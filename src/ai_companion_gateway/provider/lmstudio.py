# WIP: src/pipelines/llm/api/lmstudio.py

from ..base_handlers import BaseAPIClientWrapper
import os
import warnings
from typing import Any, BinaryIO, Optional

from typing_extensions import Buffer
from PIL import Image, ImageFile
import lmstudio as lms
from lmstudio import FileHandle

SERVER_API_HOST = "localhost:1234"

lms.get_default_client(SERVER_API_HOST)


try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class LMStudioIntegrator(BaseAPIClientWrapper):
    def __init__(self, selected_model: str = None, api_key: str = "not-needed", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        # self.lora_model_id: str | None = lora_model_id

        # if selected_model is not None:
        #     self.local_model_path = selected_model
        # else:
        #     self.local_model_path = self.model_id

        # self.local_lora_model_path = self.lora_model_id

        self.system_prompt: Optional[str] = None
        self.user_message: Optional[str] = None
        self.chat_history: Optional[list[dict[str, Any]]] = None

        self.max_tokens = self.max_length if self.max_length > 0 else 4096

        self.llm: lms.LLM | None = None
        self.chat: lms.Chat | None = None
        self.server_url = str(kwargs.get("server_url", "http://localhost:1234"))
        self.client = lms.Client(self.server_url)
        self.image_handle: lms.FileHandle | Any | None = None

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="lmstudio",
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
            self.llm = self.client.llm.model(self.model, config={"seed": self.seed})

    def generate_answer(self, history: list[dict[str, str | dict[str, str] | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            self.system_prompt = next((msg["content"] for msg in history[:1] if msg["role"] == "system"), None)

            # messages = [{"role": msg['role'], "content": msg['content']} for msg in history[(1 if self.system_prompt else 0):-1]]
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[bool(self.system_prompt) : -1]]

            self.chat = lms.Chat(self.system_prompt) if self.system_prompt else lms.Chat()

            for msg in messages:
                if msg["role"] == "user":
                    if msg["content"]["type"] == "text":
                        user_content = msg["content"]["text"]
                    elif msg["content"]["type"] == "image":
                        user_content = msg["content"]["image_url"]
                    self.chat.add_user_message(user_content)
                elif msg["role"] == "assistant":
                    if msg["content"]["type"] == "text":
                        self.chat.add_assistant_response(msg["content"]["text"])

            self.user_message = history[-1]["content"]["text"]

            if self.image_input is not None:
                if isinstance(self.image_input, list[str]):
                    self.image_handle: FileHandle = []
                    for image in self.image_input:
                        self.image_handle.append(lms.prepare_image(image))
                else:
                    self.image_handle = lms.prepare_image(self.image_input)

                self.chat.add_user_message(self.user_message, images=[self.image_handle])
            else:
                self.chat.add_user_message(self.user_message)

            if self.enable_streaming is True:
                streamer = self.llm.respond_stream(
                    self.chat,
                    config={
                        "temperature": self.temperature,
                        "topPSampling": self.top_p,
                        "topKSampling": self.top_k,
                        "repeatPenalty": self.repetition_penalty,
                        "maxTokens": self.max_tokens,
                    },
                )
                answer = ""
                for fragment in streamer:
                    print(fragment.content, end="", flush=True)
                    answer += fragment.content

            else:
                response = self.llm.respond(
                    self.chat,
                    config={
                        "temperature": self.temperature,
                        "topPSampling": self.top_p,
                        "topKSampling": self.top_k,
                        "repeatPenalty": self.repetition_penalty,
                        "maxTokens": self.max_tokens,
                    },
                )
                answer = response.content

            self.memory.add(messages=[{"role": "user", "content": self.user_message}, {"role": "assistant", "content": answer}], user_id="ai_companion_user")

            return answer.strip()
