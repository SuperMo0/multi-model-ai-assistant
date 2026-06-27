import os
import anthropic
from anthropic.types import MessageParam, Message, TextBlock
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from llm.base import LLMResponse, TokenUsage

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

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
        return (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    def _execute_call(self, messages: list[MessageParam], model: str, stream: bool) -> LLMResponse:
        if stream:
            return self._stream(messages, model)

        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=1024,
        )
        return self._parse(response, model)

    def complete(self, messages: list[MessageParam], model: str = "", stream: bool = False) -> LLMResponse:
        return self._execute_call(messages, model or self.model, stream)

    def parse(self, messages, response_type, model="", **kwargs):
        model = model or self.model
        response = self.client.messages.parse(
            model=model,
            messages=messages,
            max_tokens=2048,
            output_format=response_type,
        )
        return response.parsed_output, self._parse(response, model)

    def _parse(self, response: Message, model: str) -> LLMResponse:
        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        text = "".join(block.text for block in response.content if isinstance(block, TextBlock))

        return LLMResponse(
            text=text,
            model=model,
            tokens=TokenUsage(prompt=prompt_tokens, completion=completion_tokens, total=total_tokens),
            cost=cost,
        )

    def _stream(self, messages: list[MessageParam], model: str) -> LLMResponse:
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
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text="".join(chunks),
            model=model,
            tokens=TokenUsage(prompt=prompt_tokens, completion=completion_tokens, total=total_tokens),
            cost=cost,
        )
