from typing import Protocol, Type, TypeVar, runtime_checkable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
    def parse(self, messages: list[dict], response_type: Type[T], **kwargs) -> tuple[T, LLMResponse]: ...
