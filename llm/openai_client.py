import os
import time
import openai
from dotenv import load_dotenv

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

    def complete(self, messages: list[dict], model: str = "", stream: bool = False) -> dict:
        model = model or self.model
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if stream:
                    return self._stream(messages, model)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return self._parse(response, model)
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _parse(self, response, model: str) -> dict:
        usage = response.usage
        pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
        cost = (
            usage.prompt_tokens * pricing["input"]
            + usage.completion_tokens * pricing["output"]
        ) / 1_000_000
        return {
            "text": response.choices[0].message.content or "",
            "model": model,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
            "cost": cost,
        }

    def _stream(self, messages: list[dict], model: str) -> dict:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
        print()
        text = "".join(chunks)
        # streaming doesn't return usage, estimate from text length
        prompt_tokens = sum(len(m["content"].split()) * 4 // 3 for m in messages)
        completion_tokens = len(text.split()) * 4 // 3
        pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000
        return {
            "text": text,
            "model": model,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost": cost,
        }
