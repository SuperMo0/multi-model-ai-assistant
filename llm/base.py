from typing import Protocol, runtime_checkable
from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt: int
    completion: int
    total: int


class LLMResponse(BaseModel):
    text: str
    model: str
    tokens: TokenUsage
    cost: float


@runtime_checkable
class LLMClient(Protocol):
    model: str
    def complete(self, messages: list[dict], **kwargs) -> LLMResponse: ...
