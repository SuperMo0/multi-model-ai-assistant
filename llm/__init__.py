from llm.base import LLMClient
from llm.openai_client import OpenAIClient
from llm.anthropic_client import AnthropicClient


def get_client(provider: str) -> LLMClient:
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }
    if provider not in clients:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(clients)}")
    return clients[provider]()
