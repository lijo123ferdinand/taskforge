import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from taskforge.providers.anthropic import (AnthropicProvider, from_anthropic_response, to_anthropic_messages,
                                           to_anthropic_tools)
from taskforge.providers.base import CompletionRequest, Message, ProviderError, ToolCall, ToolSpec
from taskforge.providers.openai_compat import OpenAICompatProvider, from_openai_response, to_openai_messages

CONVO = [
    Message("user", "hi"),
    Message("assistant", "using tools", tool_calls=[ToolCall("t1", "read_file", {"path": "a"}),
                                                    ToolCall("t2", "calculator", {"expression": "1+1"})]),
    Message("tool", "A", tool_call_id="t1", name="read_file"),
    Message("tool", "ERROR: x", tool_call_id="t2", name="calculator", is_error=True),
]


def test_anthropic_merges_consecutive_tool_results_into_one_user_turn():
    out = to_anthropic_messages(CONVO)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True


def test_anthropic_provider_with_injected_client():
    seen = {}

    class Messages:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok "),
                         SimpleNamespace(type="tool_use", id="tu_1", name="calculator", input={"expression": "2"})],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7), stop_reason="tool_use", model="m")

    provider = AnthropicProvider(model="claude-sonnet-5-5", client=SimpleNamespace(messages=Messages()))
    req = CompletionRequest("sys", [Message("user", "q")], [ToolSpec("calculator", "calc", {"type": "object"})])
    comp = asyncio.run(provider.complete(req))
    assert seen["model"] == "claude-sonnet-5-5" and seen["system"] == "sys"
    assert "temperature" not in seen  # only sent when explicitly requested
    assert seen["tools"] == to_anthropic_tools(req)
    assert comp.text == "ok " and comp.tool_calls[0].arguments == {"expression": "2"}
    assert comp.usage.total == 18


def test_openai_message_shape():
    out = to_openai_messages("sys", CONVO)
    assert out[0] == {"role": "system", "content": "sys"}
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "A"}


def test_openai_unparseable_arguments_are_kept_for_schema_rejection():
    comp = from_openai_response({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c", "function": {"name": "read_file", "arguments": "{not json"}}]}, "finish_reason": "tool_calls"}]})
    assert comp.tool_calls[0].arguments == {"_unparsed_arguments": "{not json"}
    with pytest.raises(ProviderError):
        from_openai_response({"error": "nope"})


class _Handler(BaseHTTPRequestHandler):
    status = 200
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, self.headers.get("Authorization"), body))
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b"overloaded")
            return
        payload = {"model": body["model"], "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                   "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
                       {"id": "c1", "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"expression\": \"1+2\"}"}}]}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.status, _Handler.requests = 200, []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", _Handler
    srv.shutdown()


def test_openai_compat_provider_round_trip_over_real_http(server):
    url, handler = server
    provider = OpenAICompatProvider(model="llama3.1", base_url=url, api_key="test-key")
    req = CompletionRequest("sys", [Message("user", "add")], [ToolSpec("calculator", "c", {"type": "object"})])
    comp = asyncio.run(provider.complete(req))
    path, auth, body = handler.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer test-key"
    assert body["tools"][0]["function"]["name"] == "calculator"
    assert comp.tool_calls[0].arguments == {"expression": "1+2"} and comp.usage.total == 8


def test_openai_compat_5xx_is_retryable(server):
    url, handler = server
    handler.status = 503
    with pytest.raises(ProviderError) as err:
        asyncio.run(OpenAICompatProvider(base_url=url, api_key="").complete(
            CompletionRequest("s", [Message("user", "x")])))
    assert err.value.retryable
