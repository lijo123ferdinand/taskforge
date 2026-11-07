"""Provider-neutral message types and the Provider protocol.

Agents speak only these types. Each concrete provider converts them to and from
its wire format, which keeps vendor quirks (Anthropic's content blocks, OpenAI's
stringified tool arguments) out of the orchestration logic and lets the offline
MockProvider stand in for any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ToolCall",
    "Message",
    "ToolSpec",
    "CompletionRequest",
    "Usage",
    "Completion",
    "Provider",
    "ProviderError",
    "estimate_tokens",
]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A conversation turn.

    ``role="tool"`` carries a tool result; ``tool_call_id`` links it back to the
    assistant's :class:`ToolCall`. Both Anthropic and OpenAI require that link.
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for tool results
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class CompletionRequest:
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 1024
    # None = provider default. Some newer models reject an explicit temperature, so we
    # only send one when the caller asks for it.
    temperature: float | None = None
    # Which agent is calling ("planner" / "worker" / "critic"). Used for tracing and
    # cost attribution; real providers ignore it.
    agent: str = "worker"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    model: str = ""


class ProviderError(RuntimeError):
    """A provider call failed. ``retryable`` marks transient faults (429, 5xx, timeouts)."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, request: CompletionRequest) -> Completion: ...


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Used for budget pre-checks and by the mock provider.

    Deliberately crude: it only has to be in the right ballpark to stop a run
    *before* sending a prompt that would obviously blow the budget. Real usage
    numbers reported by the provider are what get charged.
    """
    return max(1, (len(text) + 3) // 4)


def request_text(request: CompletionRequest) -> str:
    """Flatten a request into one string (for token estimation)."""
    parts = [request.system]
    for m in request.messages:
        parts.append(m.content)
        for call in m.tool_calls:
            parts.append(f"{call.name}{call.arguments}")
    for t in request.tools:
        parts.append(f"{t.name}{t.description}{t.parameters}")
    return "\n".join(parts)
