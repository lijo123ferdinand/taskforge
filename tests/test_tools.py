import asyncio
import json
import os
import time

import pytest

from taskforge.tools import ToolContext, ToolError, default_registry, resolve_in_sandbox, safe_calculate


def invoke(ws, name, **args):
    return asyncio.run(default_registry().invoke(name, args, ToolContext(sandbox=ws)))


def test_write_then_read_round_trip(empty_ws):
    assert invoke(empty_ws, "write_file", path="out/a.txt", content="hello").ok
    res = invoke(empty_ws, "read_file", path="out/a.txt")
    assert res.ok and res.content == "hello"
    assert "out/a.txt (5 bytes)" in invoke(empty_ws, "list_dir").content


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../x", "~/x", ".taskforge/checkpoint.json"])
def test_sandbox_rejects_escapes_and_reserved_dir(empty_ws, path):
    res = invoke(empty_ws, "read_file", path=path)
    assert not res.ok
    assert any(w in res.content for w in ("escapes", "absolute", "reserved"))


def test_symlink_pointing_outside_is_rejected(empty_ws, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("s3cret")
    os.symlink(outside, empty_ws / "link.txt")
    with pytest.raises(ToolError, match="escapes"):
        resolve_in_sandbox(empty_ws, "link.txt")


def test_schema_validation_errors_go_back_as_tool_results(empty_ws):
    res = invoke(empty_ws, "write_file", path="a.txt")
    assert not res.ok and "missing required property 'content'" in res.content
    res = invoke(empty_ws, "read_file", path="a.txt", mode="rb")
    assert not res.ok and "unexpected property" in res.content
    res = asyncio.run(default_registry().invoke("rm_rf", {}, ToolContext(sandbox=empty_ws)))
    assert not res.ok and "unknown tool" in res.content


@pytest.mark.parametrize("expr, value", [("(12*7)+3", 87), ("sqrt(144)*5", 60), ("2**10-24", 1000),
                                         ("7/2", 3.5), ("-3 + abs(-5)", 2), ("round(pi, 2)", 3.14)])
def test_calculator(expr, value):
    assert safe_calculate(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "9**9**9", "1/0", "x + 1", "open('f')",
                                  "().__class__"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        safe_calculate(expr)


def test_python_eval_returns_result_and_stdout(csv_ws):
    code = "rows = list(csv.DictReader(read_text('data/sales.csv').splitlines()))\nprint(len(rows))\n" \
           "result = round(statistics.fmean(float(r['units']) for r in rows), 4)"
    res = invoke(csv_ws, "python_eval", code=code)
    assert res.ok, res.content
    data = json.loads(res.content)
    assert data["stdout"].strip() == "40"
    assert isinstance(data["result"], float)


@pytest.mark.parametrize("code, fragment", [
    ("import os", "import statements are not allowed"),
    ("x = ().__class__.__bases__", "attribute '__"),
    ("x = '{0.__class__}'.format(1)", "attribute 'format'"),
    ("open('x', 'w')", "NameError"),
    ("eval('1')", "NameError"),
    ("x = statistics.sys", "AttributeError"),
    ("read_text('../../etc/passwd')", "escapes the sandbox"),
    ("def f(:\n  pass", "SyntaxError"),
])
def test_python_eval_restrictions(empty_ws, code, fragment):
    res = invoke(empty_ws, "python_eval", code=code)
    assert not res.ok
    assert fragment in res.content, res.content


def test_python_eval_timeout_kills_runaway_code(empty_ws):
    start = time.monotonic()
    res = invoke(empty_ws, "python_eval", code="while True:\n    pass", timeout_seconds=0.5)
    assert not res.ok and ("timed out" in res.content or "exited" in res.content)
    assert time.monotonic() - start < 5


def test_python_eval_inputs_are_injected(empty_ws):
    res = invoke(empty_ws, "python_eval", code="result = sum(xs)", inputs={"xs": [1, 2, 3]})
    assert res.ok and json.loads(res.content)["result"] == 6


def test_web_fetch_is_not_even_registered_by_default():
    assert "web_fetch" not in default_registry().names()
    assert "web_fetch" in default_registry(allow_network=True).names()


def test_web_fetch_rejects_non_http_and_unlisted_hosts(empty_ws):
    reg = default_registry(allow_network=True)
    res = asyncio.run(reg.invoke("web_fetch", {"url": "file:///etc/passwd"}, ToolContext(sandbox=empty_ws)))
    assert not res.ok and "pattern" in res.content
    ctx = ToolContext(sandbox=empty_ws, extra={"allowed_hosts": {"example.org"}})
    res = asyncio.run(reg.invoke("web_fetch", {"url": "http://127.0.0.1:9/"}, ctx))
    assert not res.ok and "allow-list" in res.content
