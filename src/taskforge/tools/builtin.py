"""The built-in tools: sandboxed file I/O, a safe calculator, a python-eval sandbox, web fetch.

Threat model, stated plainly: these tools protect against a *confused or sloppy*
model (wrong paths, runaway loops, accidental deletes), and make a deliberate
escape noticeably harder. They are not a security boundary against a determined
attacker; for that, run the whole engine inside a container or VM.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import operator
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .base import RESERVED_DIR, Tool, ToolContext, ToolError, ToolRegistry, resolve_in_sandbox

__all__ = ["default_registry", "safe_calculate", "PYTHON_EVAL_RUNNER"]

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 1_000_000

# --------------------------------------------------------------------- files


def _read_file(ctx: ToolContext, path: str) -> str:
    target = resolve_in_sandbox(ctx.sandbox, path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    if target.stat().st_size > MAX_READ_BYTES:
        raise ToolError(f"{path} is larger than {MAX_READ_BYTES} bytes")
    return target.read_text(encoding="utf-8", errors="replace")


def _write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = resolve_in_sandbox(ctx.sandbox, path)
    data = content.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ToolError(f"content is larger than {MAX_WRITE_BYTES} bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a crash never leaves a half-written artifact behind.
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return f"wrote {len(data)} bytes to {path}"


def _list_dir(ctx: ToolContext, path: str = ".") -> str:
    base = ctx.sandbox.resolve() if path in (".", "") else resolve_in_sandbox(ctx.sandbox, path)
    if not base.is_dir():
        raise ToolError(f"not a directory: {path}")
    root = ctx.sandbox.resolve()
    entries = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root)
        if rel.parts[0] == RESERVED_DIR or not p.is_file():
            continue
        entries.append(f"{rel.as_posix()} ({p.stat().st_size} bytes)")
        if len(entries) >= 500:
            entries.append("... (truncated)")
            break
    return "\n".join(entries) or "(empty)"


# ---------------------------------------------------------------- calculator

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil, "abs": abs,
          "round": round, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_calculate(expression: str) -> float | int:
    """Evaluate arithmetic by walking the AST. Never calls ``eval``.

    Exponents are capped because ``9**9**9`` is a one-line denial of service.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid expression: {expression!r}") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e100):
                raise ToolError("exponent too large")
            try:
                return _BIN[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError(f"unsupported syntax: {ast.dump(node)[:80]}")

    result = ev(tree)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return result


def _calculator(ctx: ToolContext, expression: str) -> str:
    value = safe_calculate(expression)
    return str(round(value, 10) if isinstance(value, float) else value)


# --------------------------------------------------------------- python_eval

# The runner executes in a *separate* interpreter started with -I (isolated: no
# user site-packages, no PYTHON* env vars, script dir not on sys.path). Layers:
#   1. AST screen: no imports statements, no dunder/frame attributes, no
#      str.format (it can reach attributes through format strings).
#   2. Restricted builtins: no open/exec/eval/compile/getattr/__import__...
#   3. Whitelisted modules exposed as read-only namespaces of their public,
#      non-module attributes (statistics.sys would otherwise leak sys).
#   4. OS limits: CPU seconds, address space (Linux), and a wall-clock timeout
#      enforced by the parent, which kills the process.
PYTHON_EVAL_RUNNER = r'''
import ast, json, sys, types, io, contextlib, pathlib
payload = json.loads(sys.stdin.read())
try:
    import resource
    cpu = int(payload["cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    try:
        mem = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass  # macOS does not support RLIMIT_AS; CPU limit + timeout still apply
except ImportError:
    pass
BLOCKED_ATTRS = {"format", "format_map", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame",
                 "f_globals", "f_locals", "f_back", "f_builtins", "tb_frame", "mro"}
def fail(message):
    print(json.dumps({"error": message}))
    raise SystemExit(0)
try:
    tree = ast.parse(payload["code"], mode="exec")
except SyntaxError as exc:
    fail("SyntaxError: %s (line %s)" % (exc.msg, exc.lineno))
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        fail("import statements are not allowed; whitelisted modules are pre-imported: " + ", ".join(payload["modules"]))
    if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRS):
        fail("access to attribute %r is not allowed" % node.attr)
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        fail("access to name %r is not allowed" % node.id)
root = pathlib.Path(payload["sandbox"]).resolve()
def read_text(path):
    target = (root / str(path)).resolve()
    if not target.is_relative_to(root) or pathlib.Path(str(path)).is_absolute():
        raise PermissionError("path escapes the sandbox: %r" % (path,))
    if target.relative_to(root).parts[:1] == (".taskforge",):
        raise PermissionError(".taskforge/ is reserved")
    return target.read_text(encoding="utf-8")
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
                 "int", "isinstance", "len", "list", "map", "max", "min", "print", "range", "repr", "reversed",
                 "round", "set", "sorted", "str", "sum", "tuple", "zip", "ValueError", "KeyError",
                 "TypeError", "ZeroDivisionError", "Exception", "IndexError", "StopIteration", "True", "False", "None"]
import builtins
safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)}
def expose(mod):
    ns = {k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, types.ModuleType)}
    return types.SimpleNamespace(**ns)
env = {"__builtins__": safe, "read_text": read_text}
for name in payload["modules"]:
    env[name] = expose(__import__(name))
env.update(payload.get("inputs") or {})
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<python_eval>", "exec"), env)
except BaseException as exc:
    print(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc), "stdout": buf.getvalue()[-4000:]}))
    raise SystemExit(0)
out = {"stdout": buf.getvalue()[-20000:]}
if "result" in env:
    try:
        json.dumps(env["result"])
        out["result"] = env["result"]
    except (TypeError, ValueError):
        out["result"] = repr(env["result"])
print(json.dumps(out))
'''

ALLOWED_MODULES = ("math", "statistics", "json", "re", "csv", "collections", "itertools", "functools", "string")


async def _python_eval(ctx: ToolContext, code: str, timeout_seconds: float = 5.0,
                       inputs: dict[str, Any] | None = None) -> str:
    timeout = float(min(max(timeout_seconds, 0.1), 30.0))
    payload = json.dumps({"code": code, "inputs": inputs or {}, "sandbox": str(ctx.sandbox.resolve()),
                          "modules": list(ALLOWED_MODULES), "cpu_seconds": max(1, math.ceil(timeout))})
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", PYTHON_EVAL_RUNNER,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=str(ctx.sandbox), env={"PYTHONIOENCODING": "utf-8"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"timed out after {timeout:g}s") from None
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        # Killed by an OS limit (e.g. SIGXCPU) or crashed before reporting.
        raise ToolError(f"interpreter exited with code {proc.returncode}: {stderr.decode(errors='replace')[-500:]}")
    # The runner's protocol is "last stdout line is JSON"; user prints are captured, not leaked.
    last = text.splitlines()[-1]
    try:
        data = json.loads(last)
    except json.JSONDecodeError as exc:
        raise ToolError(f"unexpected sandbox output: {last[:300]}") from exc
    if "error" in data:
        raise ToolError(data["error"] + (f"\nstdout so far:\n{data['stdout']}" if data.get("stdout") else ""))
    return json.dumps(data)


# ------------------------------------------------------------------- web

async def _web_fetch(ctx: ToolContext, url: str, max_bytes: int = 100_000) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError("only http(s) URLs are allowed")
    allowed = ctx.extra.get("allowed_hosts")
    if allowed is not None and parsed.hostname not in allowed:
        raise ToolError(f"host {parsed.hostname!r} is not in the allow-list")

    def fetch() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "taskforge/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read(max_bytes).decode("utf-8", errors="replace")

    return await asyncio.to_thread(fetch)


# --------------------------------------------------------------- registry

def default_registry(*, allow_network: bool = False) -> ToolRegistry:
    """All built-in tools. ``web_fetch`` is only registered when network is explicitly allowed,
    so a model cannot even *see* it by default (an unseen tool cannot be prompt-injected into use)."""
    path = {"type": "string", "minLength": 1, "maxLength": 512}
    tools = [
        Tool("read_file", "Read a UTF-8 text file from the workspace.",
             {"type": "object", "properties": {"path": path}, "required": ["path"], "additionalProperties": False},
             _read_file),
        Tool("write_file", "Create or overwrite a UTF-8 text file in the workspace.",
             {"type": "object", "properties": {"path": path, "content": {"type": "string"}},
              "required": ["path", "content"], "additionalProperties": False},
             _write_file),
        Tool("list_dir", "List files (recursively) under a workspace directory.",
             {"type": "object", "properties": {"path": path}, "additionalProperties": False},
             _list_dir),
        Tool("calculator", "Evaluate an arithmetic expression (+ - * / // % **, sqrt, log, exp, pi...).",
             {"type": "object", "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 500}},
              "required": ["expression"], "additionalProperties": False},
             _calculator),
        Tool("python_eval",
             "Run a short Python snippet in a restricted sandbox. No imports; these modules are pre-loaded: "
             + ", ".join(ALLOWED_MODULES) + ". read_text(path) reads workspace files. Assign `result` to return "
             "a JSON value; print() output is captured.",
             {"type": "object", "properties": {
                 "code": {"type": "string", "minLength": 1, "maxLength": 20000},
                 "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 30},
                 "inputs": {"type": "object"}},
              "required": ["code"], "additionalProperties": False},
             _python_eval),
    ]
    if allow_network:
        tools.append(Tool("web_fetch", "Fetch a URL over HTTP(S) and return its text (truncated).",
                          {"type": "object", "properties": {
                              "url": {"type": "string", "pattern": "^https?://"},
                              "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1_000_000}},
                           "required": ["url"], "additionalProperties": False},
                          _web_fetch, requires_network=True))
    return ToolRegistry(tools)


def sandbox_manifest(root: Path, limit: int = 200) -> list[str]:
    """``path (N bytes)`` lines for every workspace file; given to the planner as context.

    CSV files also get their header row: a planner that knows the columns can write
    precise acceptance checks ("stats must cover price and units") instead of vague ones.
    """
    lines = _list_dir(ToolContext(sandbox=root), ".").splitlines()[:limit]
    out = []
    for line in lines:
        rel = line.rsplit(" (", 1)[0]
        if rel.endswith(".csv"):
            try:
                with (root / rel).open(encoding="utf-8", errors="replace") as fh:
                    header = fh.readline().strip()
                line = f"{line} header: {header[:300]}"
            except OSError:
                pass
        out.append(line)
    return out
