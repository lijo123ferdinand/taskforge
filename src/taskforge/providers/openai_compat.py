"""OpenAI-compatible Chat Completions provider over plain ``urllib``.

Works with OpenAI, Ollama (``http://localhost:11434/v1``), vLLM, LM Studio and any
other server speaking the ``/chat/completions`` dialect with function tools. No
SDK on purpose: one HTTP POST does not justify a dependency.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import Completion, CompletionRequest, Message, ProviderError, ToolCall, Usage

__all__ = ["OpenAICompatProvider", "to_openai_messages", "from_openai_response"]

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     # The OpenAI wire format wants arguments as a JSON *string*.
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in m.tool_calls
                ]
            out.append(msg)
        else:
            out.append({"role": "user", "content": m.content})
    return out


def from_openai_response(data: dict[str, Any]) -> Completion:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed chat completion: {str(data)[:200]}") from exc
    calls: list[ToolCall] = []
    for i, raw in enumerate(msg.get("tool_calls") or []):
        fn = raw.get("function", {})
        args_raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except json.JSONDecodeError:
            # Keep the garbage visible: schema validation will reject it and the error
            # goes back to the model, which is usually enough for it to self-correct.
            args = {"_unparsed_arguments": args_raw}
        if not isinstance(args, dict):
            args = {"_unparsed_arguments": args_raw}
        calls.append(ToolCall(id=raw.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=args))
    usage = data.get("usage") or {}
    return Completion(
        text=msg.get("content") or "",
        tool_calls=calls,
        usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
        stop_reason=str(choice.get("finish_reason") or ""),
        model=str(data.get("model", "")),
    )


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, model: str = "llama3.1", base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            raise ProviderError(f"HTTP {exc.code}: {body}", retryable=exc.code in (408, 429) or exc.code >= 500) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"connection failed: {exc}", retryable=True) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("response was not JSON") from exc

    async def complete(self, request: CompletionRequest) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(request.system, request.messages),
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}}
                for t in request.tools
            ]
        # urllib is blocking; run it on a worker thread so other agents keep going.
        data = await asyncio.to_thread(self._post, payload)
        return from_openai_response(data)
