import warnings
from typing import Any, Optional

from PIL import Image, ImageFile

from openai import OpenAI
from openai.types.responses.response import Response

from pydantic import SecretStr

from ai_companion_core import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator

    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn(
        "langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ",
        UserWarning,
    )
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False


class OpenAIClientWrapper(BaseAPIClientWrapper):
    def __init__(
        self,
        selected_model: str,
        api_key: str | SecretStr | None = None,
        use_langchain: bool = True,
        image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None,
        **kwargs,
    ):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        # self.client = OpenAI(api_key=self.api_key)
        self.system_prompt: Optional[str | Any] = None

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="openai",
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
                enable_thinking=self.enable_thinking,
                image_input=self.image_input,
            )
        else:
            self.client = OpenAI(api_key=self.api_key)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            # client = OpenAI(api_key=self.api_key)
            # openai.api_key = self.api_key

            self.system_prompt = next((msg["content"] for msg in history[:1] if msg["role"] == "system"), None)

            # --- Tool calling path (OpenAI Responses API) ---
            if self.can_run_tools():
                from ..tool_calling import run_tool_loop_responses

                # Feed the system prompt via `instructions`, others as input items.
                tool_input = [{"role": msg["role"], "content": self.content_to_openai(msg["content"])} for msg in history if msg["role"] != "system"]
                non_reasoning = "none" if "gpt-5.1" in self.model else "minimum"
                answer = run_tool_loop_responses(
                    client=self.client,
                    model=self.model,
                    input_items=tool_input,
                    tool_specs=self.get_tool_specs(),
                    executor=self.tool_executor,
                    instructions=self.system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    max_iterations=self.max_tool_iterations,
                    reasoning={"effort": "medium"} if self.enable_thinking else {"effort": non_reasoning},
                    extra_body={
                        "frequency_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                        "presence_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                    },
                )
                return answer.strip()

            # if self.system_prompt is not None:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[1:-1]]
            # else:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]

            # messages = [{"role": msg['role'], "content": msg['content']} for msg in history[(1 if self.system_prompt else 0):-1]]
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[bool(self.system_prompt) : -1]]

            if self.image_input is not None:
                with open(self.image_input, "rb") as f:
                    image = self.client.files.create(file=f, purpose="vision")

                new_message = {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": history[-1]["content"]["text"]},
                        {"type": "input_image", "file_id": image.id},
                    ],
                }
            else:
                new_message = {"role": "user", "content": history[-1]["content"]}

            messages.append(new_message)

            logger.info(f"[*] OpenAI API 요청: {messages}")

            # response = client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     top_logprobs=self.top_k,
            #     top_p=self.top_p,
            #     frequency_penalty=self.repetition_penalty,
            #     presence_penalty=self.repetition_penalty,
            # )

            # answer = response.choices[0].message["content"]

            non_reasoning = "none" if "gpt-5.1" in self.model else "minimum"

            if self.enable_streaming is True:
                answer = ""
                with self.client.responses.stream(
                    model=self.model,
                    input=messages,
                    instructions=self.system_prompt,
                    store=False,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_logprobs=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    reasoning={"effort": "medium"} if self.enable_thinking else {"effort": non_reasoning},
                    extra_body={
                        "frequency_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                        "presence_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                    },
                ) as stream:
                    for event in stream:
                        print(event.text, end="", flush=True)
                        answer += event.text

            else:
                response: Response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    instructions=self.system_prompt,
                    store=False,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_logprobs=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    reasoning={"effort": "medium"} if self.enable_thinking else {"effort": non_reasoning},
                    extra_body={
                        "frequency_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                        "presence_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                    },
                )
                answer = response.output_text

            return answer.strip()


# class OpenAILangChainIntegration:
#     def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.6, top_p=0.9, top_k=40, repetition_penalty=1.0):
#         self.api_key = api_key
#         if not api_key:
#             logger.error("OpenAI API Key가 missing.")
#             raise "OpenAI API Key가 필요합니다."

#         self.model = model
#         self.llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             top_p=top_p,
#             top_logprobs=top_k,
#             frequency_penalty=repetition_penalty,
#             api_key=api_key,
#             max_tokens=2048
#         )
