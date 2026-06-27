import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
CAPABLE_MODEL = "claude-sonnet-4-5"


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict], model: str = "", stream: bool = False) -> dict:
        model = model or self.model
        try:
            if stream:
                return self._stream(messages, model)
            response = self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )
            return self._parse(response, model)
        except anthropic.RateLimitError:
            return {}

    def _parse(self, response, model: str) -> dict:
        usage = response.usage
        pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
        cost = (
            usage.input_tokens * pricing["input"]
            + usage.output_tokens * pricing["output"]
        ) / 1_000_000
        return {
            "text": response.content,
            "model": model,
            "tokens": {
                "prompt": usage.input_tokens,
                "completion": usage.output_tokens,
                "total": usage.input_tokens + usage.output_tokens,
            },
            "cost": cost,
        }

    def _stream(self, messages: list[dict], model: str) -> dict:
        with self.client.messages.stream(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=256,
        ) as stream:
            chunks = []
            for text in stream.text_stream:
                print(text, end="", flush=True)
                chunks.append(text)
            print()
            return {
                "text": "".join(chunks),
                "model": model,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost": 0.0,
            }
