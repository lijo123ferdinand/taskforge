"""Anthropic Messages API provider (optional dependency: ``pip install taskforge[anthropic]``).

The SDK is imported lazily so the core package stays dependency-free and the
import error only appears if you actually select this provider.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Completion, CompletionRequest, Message, ProviderError, ToolCall, Usage

__all__ = ["AnthropicProvider", "to_anthropic_messages", "to_anthropic_tools", "from_anthropic_response"]

DEFAULT_MODEL = "claude-sonnet-5-5"


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Anthropic's content-block format.

    Anthropic requires all tool results answering one assistant turn to arrive in a
    *single* user message, so consecutive ``tool`` messages are merged.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
                "is_error": m.is_error,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def to_anthropic_tools(request: CompletionRequest) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools]


def from_anthropic_response(resp: Any) -> Completion:
    """Accepts an SDK response object (attribute access) and normalises it."""
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in resp.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(block.text)
        elif kind == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
    usage = Usage(int(resp.usage.input_tokens), int(resp.usage.output_tokens))
    return Completion(text="".join(texts), tool_calls=calls, usage=usage,
                      stop_reason=str(resp.stop_reason or ""), model=str(getattr(resp, "model", "")))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, client: Any = None) -> None:
        self.model = model
        if client is not None:
            self._client = client  # injectable for tests
            return
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError("the anthropic provider needs `pip install taskforge[anthropic]`") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._sdk = anthropic

    async def complete(self, request: CompletionRequest) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": to_anthropic_messages(request.messages),
        }
        if request.tools:
            kwargs["tools"] = to_anthropic_tools(request)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK raises its own hierarchy; normalise retryability
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 429, 500, 502, 503, 504, 529) or "timeout" in type(exc).__name__.lower() \
                or "connection" in type(exc).__name__.lower()
            raise ProviderError(f"anthropic call failed: {exc}", retryable=retryable) from exc
        return from_anthropic_response(resp)
