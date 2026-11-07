"""Task definitions and the validated task DAG.

The planner is an LLM, so its output is *untrusted input*. Everything a model can
get wrong about a plan is checked here before a single worker runs:

* duplicate task ids,
* dependencies on tasks that do not exist,
* self-dependencies and longer cycles (which would deadlock the scheduler),
* malformed fields (wrong types, unknown check kinds).

The error messages are written to be fed back to the planner for a re-plan, so
they name the offending task and, for cycles, print the actual cycle path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["PlanError", "Task", "TaskGraph", "CHECK_KINDS"]

# Deterministic acceptance checks the critic runs before (and instead of, when they
# fail) an LLM judgement. Keep this list in sync with taskforge.checks.
CHECK_KINDS = frozenset({"contains", "not_contains", "regex", "min_length", "json_keys", "file_exists"})

_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


class PlanError(ValueError):
    """The plan is structurally invalid (cycle, missing dependency, bad field...)."""


@dataclass(frozen=True)
class Task:
    """One node of the plan.

    ``acceptance`` holds natural-language criteria for the LLM critic; ``checks``
    holds machine-checkable ones (see ``CHECK_KINDS``). ``tools`` narrows the
    role's tool allow-list for this task: a task can never gain a tool its role
    does not already allow.
    """

    id: str
    description: str
    role: str = "generalist"
    deps: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None
    acceptance: tuple[str, ...] = ()
    checks: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        if not isinstance(data, dict):
            raise PlanError(f"each task must be an object, got {type(data).__name__}")
        tid = data.get("id")
        if not isinstance(tid, str) or not _ID_RE.match(tid):
            raise PlanError(f"task id {tid!r} must match {_ID_RE.pattern}")
        desc = data.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise PlanError(f"task {tid!r}: description must be a non-empty string")

        def str_list(key: str) -> tuple[str, ...]:
            value = data.get(key, [])
            if value is None:
                return ()
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise PlanError(f"task {tid!r}: {key} must be a list of strings")
            return tuple(value)

        checks = data.get("checks", []) or []
        if not isinstance(checks, list):
            raise PlanError(f"task {tid!r}: checks must be a list")
        for check in checks:
            if not isinstance(check, dict) or check.get("type") not in CHECK_KINDS:
                raise PlanError(f"task {tid!r}: unknown check {check!r}; allowed types: {sorted(CHECK_KINDS)}")

        tools = data.get("tools")
        role = data.get("role", "generalist")
        if not isinstance(role, str):
            raise PlanError(f"task {tid!r}: role must be a string")
        return cls(
            id=tid,
            description=desc.strip(),
            role=role,
            deps=str_list("deps"),
            tools=None if tools is None else str_list("tools"),
            acceptance=str_list("acceptance"),
            checks=tuple(dict(c) for c in checks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "role": self.role,
            "deps": list(self.deps),
            "tools": None if self.tools is None else list(self.tools),
            "acceptance": list(self.acceptance),
            "checks": [dict(c) for c in self.checks],
        }


@dataclass
class TaskGraph:
    """An immutable-after-validation DAG of tasks, in the order the planner listed them."""

    tasks: dict[str, Task] = field(default_factory=dict)

    @classmethod
    def from_tasks(cls, tasks: Iterable[Task], known_roles: Iterable[str] | None = None) -> "TaskGraph":
        graph: dict[str, Task] = {}
        for task in tasks:
            if task.id in graph:
                raise PlanError(f"duplicate task id {task.id!r}")
            graph[task.id] = task
        if not graph:
            raise PlanError("plan has no tasks")

        roles = set(known_roles) if known_roles is not None else None
        for task in graph.values():
            if roles is not None and task.role not in roles:
                raise PlanError(f"task {task.id!r}: unknown role {task.role!r}; known roles: {sorted(roles)}")
            for dep in task.deps:
                if dep == task.id:
                    raise PlanError(f"task {task.id!r} depends on itself")
                if dep not in graph:
                    raise PlanError(f"task {task.id!r} depends on unknown task {dep!r}")
        out = cls(graph)
        out._check_acyclic()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], known_roles: Iterable[str] | None = None) -> "TaskGraph":
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
            raise PlanError('plan must be an object of the form {"tasks": [...]}')
        return cls.from_tasks((Task.from_dict(t) for t in data["tasks"]), known_roles)

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": [t.to_dict() for t in self.tasks.values()]}

    # ------------------------------------------------------------------ structure

    def _check_acyclic(self) -> None:
        # Iterative three-colour DFS: recursion would blow the stack on a long
        # (model-generated, possibly adversarial) chain, and we want the cycle path.
        white, grey, black = 0, 1, 2
        colour = {tid: white for tid in self.tasks}
        for root in self.tasks:
            if colour[root] != white:
                continue
            stack: list[tuple[str, int]] = [(root, 0)]
            path: list[str] = [root]
            colour[root] = grey
            while stack:
                node, idx = stack[-1]
                deps = self.tasks[node].deps
                if idx < len(deps):
                    stack[-1] = (node, idx + 1)
                    nxt = deps[idx]
                    if colour[nxt] == grey:
                        cycle = path[path.index(nxt):] + [nxt]
                        raise PlanError("dependency cycle: " + " -> ".join(cycle))
                    if colour[nxt] == white:
                        colour[nxt] = grey
                        stack.append((nxt, 0))
                        path.append(nxt)
                else:
                    colour[node] = black
                    stack.pop()
                    path.pop()

    def dependents(self, task_id: str) -> list[str]:
        return [t.id for t in self.tasks.values() if task_id in t.deps]

    def descendants(self, task_id: str) -> set[str]:
        """Everything that transitively depends on ``task_id`` (used to skip work after a failure)."""
        seen: set[str] = set()
        frontier = [task_id]
        while frontier:
            for child in self.dependents(frontier.pop()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def ready(self, done: set[str], started: set[str]) -> list[str]:
        """Tasks whose dependencies are all done and that have not been started, in plan order."""
        return [
            t.id for t in self.tasks.values()
            if t.id not in started and t.id not in done and all(d in done for d in t.deps)
        ]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; ties broken by plan order so output is stable across runs."""
        indegree = {tid: len(t.deps) for tid, t in self.tasks.items()}
        order: list[str] = []
        queue = [tid for tid in self.tasks if indegree[tid] == 0]
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for child in self.dependents(tid):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        return order

    def levels(self) -> list[list[str]]:
        """Group tasks by depth: everything in one level can run concurrently."""
        depth: dict[str, int] = {}
        for tid in self.topological_order():
            deps = self.tasks[tid].deps
            depth[tid] = 1 + max((depth[d] for d in deps), default=-1)
        out: list[list[str]] = [[] for _ in range(max(depth.values()) + 1)]
        for tid, d in depth.items():
            out[d].append(tid)
        return out

    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.tasks

    def __getitem__(self, task_id: str) -> Task:
        return self.tasks[task_id]
