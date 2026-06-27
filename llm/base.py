from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    model: str
    def complete(self, messages: list[dict], **kwargs) -> dict: ...
