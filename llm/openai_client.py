import os
import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam,ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from llm.base import LLMResponse, TokenUsage
load_dotenv()

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
DEFAULT_MODEL = "gpt-4o-mini"
CAPABLE_MODEL = "gpt-4o"



class OpenAIClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4)
    )
    def _execute_call(self, messages: list[ChatCompletionMessageParam], model: str, stream: bool) -> LLMResponse:
        if stream:
            return self._stream(messages, model)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return self._parse(response, model)

    def complete(self, messages: list[ChatCompletionMessageParam], model: str = "", stream: bool = False) -> LLMResponse:
        return self._execute_call(messages, model or self.model, stream)

    def parse(self, messages, response_type, model="", **kwargs):
        model = model or self.model
        completion = self.client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_type,
        )
        return completion.choices[0].message.parsed, self._parse(completion, model)

    def _parse(self, response: ChatCompletion, model: str) -> LLMResponse:
        usage = response.usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0

        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
            cost = (
                prompt_tokens * pricing["input"]
                + completion_tokens * pricing["output"]
            ) / 1_000_000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=model,
            tokens=TokenUsage(prompt=prompt_tokens, completion=completion_tokens, total=total_tokens),
            cost=cost,
        )

    def _stream(self, messages: list[ChatCompletionMessageParam], model: str) -> LLMResponse:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={
                "include_usage":True
            }
        )
        chunks = []
        final_usage=None
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
            if chunk.usage is not None:
                final_usage=chunk.usage

        print()
        text = "".join(chunks)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0
        if final_usage is not None:
            prompt_tokens = final_usage.prompt_tokens
            completion_tokens = final_usage.completion_tokens
            total_tokens = final_usage.total_tokens

            pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
            cost = (
                prompt_tokens * pricing["input"]
                + completion_tokens * pricing["output"]
            ) / 1_000_000

        return LLMResponse(
            text=text,
            model=model,
            tokens=TokenUsage(prompt=prompt_tokens, completion=completion_tokens, total=total_tokens),
            cost=cost,
        )
