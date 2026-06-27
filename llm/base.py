from typing import Protocol, TypedDict, runtime_checkable


class TokenUsage(TypedDict):
    prompt: int
    completion: int
    total: int


class LLMResponse(TypedDict):
    text: str
    model: str
    tokens: TokenUsage
    cost: float


@runtime_checkable
class LLMClient(Protocol):
    model: str
    def complete(self, messages: list[dict], **kwargs) -> LLMResponse: ...
