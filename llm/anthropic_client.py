import os
import anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from llm.base import LLMResponse

load_dotenv()

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
CAPABLE_MODEL = "claude-sonnet-4-5"


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    def _execute_call(self, messages: list[dict], model: str, stream: bool) -> LLMResponse:
        if stream:
            return self._stream(messages, model)
        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=1024,
        )
        return self._parse(response, model)

    def complete(self, messages: list[dict], model: str = "", stream: bool = False) -> LLMResponse:
        return self._execute_call(messages, model or self.model, stream)

    def _parse(self, response, model: str) -> LLMResponse:
        usage = response.usage
        pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
        cost = (
            usage.input_tokens * pricing["input"]
            + usage.output_tokens * pricing["output"]
        ) / 1_000_000
        return {
            "text": "".join(block.text for block in response.content if isinstance(block, TextBlock)),
            "model": model,
            "tokens": {
                "prompt": usage.input_tokens,
                "completion": usage.output_tokens,
                "total": usage.input_tokens + usage.output_tokens,
            },
            "cost": cost,
        }

    def _stream(self, messages: list[dict], model: str) -> LLMResponse:
        with self.client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=1024,
        ) as stream:
            chunks = []
            for text in stream.text_stream:
                print(text, end="", flush=True)
                chunks.append(text)
            print()
            final_message = stream.get_final_message()
            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
            cost = (
                prompt_tokens * pricing["input"]
                + completion_tokens * pricing["output"]
            ) / 1_000_000
        return {
            "text": "".join(chunks),
            "model": model,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "cost": cost,
        }
