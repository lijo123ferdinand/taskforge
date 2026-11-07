"""Tool definitions, the sandbox path guard, and the registry that executes calls.

Every tool call from a model goes through :meth:`ToolRegistry.invoke`, which is the
single choke point for: argument validation against the JSON schema, conversion
of sync tools to threads (so a slow file read never stalls the event loop), and
turning *every* failure into a tool result the model can read. A model that
sends bad arguments should get an error message and a chance to retry, not crash
the run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..providers.base import ToolSpec
from ..schema import SchemaError, validate

__all__ = ["Tool", "ToolContext", "ToolError", "ToolResult", "ToolRegistry", "resolve_in_sandbox", "RESERVED_DIR"]

# Engine state (trace, checkpoint) lives here inside the workspace. Tools must not
# be able to read or rewrite it: a model editing its own checkpoint would be a fun bug.
RESERVED_DIR = ".taskforge"


class ToolError(Exception):
    """A tool failed in an expected way; the message is shown to the model."""


@dataclass
class ToolContext:
    sandbox: Path
    task_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    content: str

    def to_text(self) -> str:
        return self.content if self.ok else f"ERROR: {self.content}"


ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    # Tools with side effects outside the process (network) are opt-in per run.
    requires_network: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


def resolve_in_sandbox(root: Path, relpath: str) -> Path:
    """Resolve ``relpath`` under ``root`` and refuse anything that escapes it.

    ``Path.resolve`` follows symlinks, so a symlink inside the sandbox pointing at
    ``/etc`` is caught too. Absolute paths are rejected outright rather than
    silently re-rooted, because silently "fixing" them hides model mistakes.
    """
    if not isinstance(relpath, str) or not relpath.strip():
        raise ToolError("path must be a non-empty string")
    if Path(relpath).is_absolute() or relpath.startswith("~"):
        raise ToolError(f"absolute paths are not allowed: {relpath!r}; use a path relative to the workspace")
    root = root.resolve()
    target = (root / relpath).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"path escapes the sandbox: {relpath!r}")
    rel = target.relative_to(root)
    if rel.parts and rel.parts[0] == RESERVED_DIR:
        raise ToolError(f"{RESERVED_DIR}/ is reserved for the engine")
    return target


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        chosen = self._tools if names is None else {n: self._tools[n] for n in names if n in self._tools}
        return [t.spec() for t in chosen.values()]

    async def invoke(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool {name!r}; available: {sorted(self._tools)}")
        try:
            validate(arguments, tool.parameters)
        except SchemaError as exc:
            return ToolResult(False, f"invalid arguments for {name}: {exc}")
        try:
            if inspect.iscoroutinefunction(tool.fn):
                value = await tool.fn(ctx, **arguments)
            else:
                value = await asyncio.to_thread(tool.fn, ctx, **arguments)
        except ToolError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:  # a buggy tool must not take the run down
            return ToolResult(False, f"{type(exc).__name__}: {exc}")
        return ToolResult(True, _render(value))
