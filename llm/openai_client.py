import os
import openai
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
    def _execute_call(self, messages: list[dict], model: str, stream: bool) -> dict:
        if stream:
            return self._stream(messages, model)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return self._parse(response, model)

    def complete(self, messages: list[dict], model: str = "", stream: bool = False) -> dict:
        return self._execute_call(messages, model or self.model, stream)

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
