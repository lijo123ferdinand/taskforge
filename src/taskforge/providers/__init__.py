from .base import (Completion, CompletionRequest, Message, Provider, ProviderError, ToolCall, ToolSpec, Usage,
                   estimate_tokens)
from .mock import CrashAfter, MockProvider, SimulatedCrash

__all__ = ["Completion", "CompletionRequest", "Message", "Provider", "ProviderError", "ToolCall", "ToolSpec",
           "Usage", "estimate_tokens", "MockProvider", "CrashAfter", "SimulatedCrash", "make_provider"]


def make_provider(name: str, model: str | None = None, **kwargs: object) -> Provider:
    """Factory used by the CLI. Real providers import lazily so `mock` needs nothing installed."""
    if name == "mock":
        return MockProvider(**kwargs)  # type: ignore[arg-type]
    if name == "anthropic":
        from .anthropic import DEFAULT_MODEL, AnthropicProvider
        return AnthropicProvider(model=model or DEFAULT_MODEL)
    if name in ("openai", "ollama"):
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model=model or "llama3.1")
    raise ValueError(f"unknown provider {name!r}; choose mock, anthropic or openai")
