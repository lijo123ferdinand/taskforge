"""The orchestrator: plan -> schedule the DAG with bounded concurrency -> review -> checkpoint.

Scheduling model
----------------
A single coroutine owns all scheduling state (task statuses, the blackboard
writes for results, checkpoints). Workers run as asyncio tasks and only *return*
results; they never mutate shared state. That is what makes checkpointing
trivially consistent: the checkpoint is written by the same coroutine that
applies results, between ``asyncio.wait`` wake-ups, so it can never capture a
half-applied state.

Concurrency is bounded by simply not launching more than ``max_concurrency``
tasks; a semaphore would also work, but then tasks would be "started" (and
traced as running) while they actually wait in a queue, which makes the
timeline lie.

Failure model
-------------
* A task that fails all revisions is ``failed``; everything downstream of it is
  ``skipped`` (it can never get its inputs) but independent branches continue.
* ``BudgetExceeded`` is a hard stop: in-flight tasks are cancelled and put back to
  ``pending`` so ``resume`` (with a bigger budget) redoes exactly those.
* A ``BaseException`` that is not an ``Exception`` (KeyboardInterrupt, a simulated
  crash) propagates untouched. The last checkpoint, written after the previous
  completed task, is what ``resume`` picks up.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import Critic, LLMGateway, Planner, Worker, WorkerFailed
from .approval import Approver, auto_approve
from .blackboard import Blackboard
from .budget import Budget, BudgetExceeded
from .checkpoint import load_checkpoint, save_checkpoint
from .dag import TaskGraph
from .providers.base import Provider, ProviderError
from .roles import DEFAULT_ROLES, Role
from .tools.base import RESERVED_DIR, ToolRegistry
from .tools.builtin import default_registry, sandbox_manifest
from .trace import Tracer

__all__ = ["RunConfig", "TaskState", "RunResult", "Orchestrator"]


@dataclass
class RunConfig:
    max_concurrency: int = 4
    use_critic: bool = True
    max_revisions: int = 2          # extra attempts after the first, when the critic rejects
    max_worker_steps: int = 8       # model turns per attempt (tool loops are bounded)
    require_approval: frozenset[str] = frozenset()
    trace_path: Path | None = None  # default: <workspace>/.taskforge/trace.jsonl
    checkpoint_path: Path | None = None  # default: <workspace>/.taskforge/checkpoint.json

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.require_approval = frozenset(self.require_approval)


@dataclass
class TaskState:
    status: str = "pending"  # pending | running | done | failed | skipped
    attempts: int = 0
    output: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "attempts": self.attempts, "output": self.output, "error": self.error}


@dataclass
class RunResult:
    run_id: str
    goal: str
    status: str  # completed | partial | failed | budget_exhausted
    graph: TaskGraph
    states: dict[str, TaskState]
    usage: dict[str, Any]
    elapsed_seconds: float
    trace_path: Path
    checkpoint_path: Path
    error: str | None = None
    blackboard: Blackboard = field(default_factory=Blackboard)

    @property
    def outputs(self) -> dict[str, str]:
        return {tid: s.output for tid, s in self.states.items() if s.status == "done" and s.output is not None}


class Orchestrator:
    def __init__(self, provider: Provider, workspace: Path | str, *, config: RunConfig | None = None,
                 registry: ToolRegistry | None = None, roles: dict[str, Role] | None = None,
                 approver: Approver = auto_approve, budget: Budget | None = None,
                 tool_extra: dict[str, Any] | None = None) -> None:
        self.provider = provider
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise FileNotFoundError(f"workspace {self.workspace} does not exist")
        self.config = config or RunConfig()
        self.registry = registry or default_registry()
        self.roles = roles or DEFAULT_ROLES
        self.approver = approver
        self.budget = budget or Budget()
        self.tool_extra = tool_extra or {}
        state_dir = self.workspace / RESERVED_DIR
        self.trace_path = Path(self.config.trace_path or state_dir / "trace.jsonl")
        self.checkpoint_path = Path(self.config.checkpoint_path or state_dir / "checkpoint.json")

    # ---------------------------------------------------------------- public

    async def run(self, goal: str) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        # A fresh run starts a fresh trace; only resume appends.
        self.trace_path.unlink(missing_ok=True)
        tracer = Tracer(self.trace_path, run_id)
        gateway = LLMGateway(self.provider, self.budget, tracer)
        start = time.monotonic()
        tracer.emit("run_started", goal=goal, provider=getattr(self.provider, "name", "?"),
                    max_concurrency=self.config.max_concurrency, use_critic=self.config.use_critic)
        try:
            graph = await Planner(gateway, self.roles, self.registry).plan(goal, sandbox_manifest(self.workspace))
        except BudgetExceeded as exc:
            tracer.emit("budget_exceeded", where="planner", error=str(exc))
            tracer.emit("run_finished", status="budget_exhausted")
            tracer.close()
            return RunResult(run_id, goal, "budget_exhausted", TaskGraph(), {}, self.budget.to_dict(),
                             time.monotonic() - start, self.trace_path, self.checkpoint_path, str(exc))
        states = {tid: TaskState() for tid in graph.tasks}
        return await self._execute(run_id, goal, graph, states, Blackboard(), gateway, start)

    async def resume(self, *, retry_failed: bool = False) -> RunResult:
        data = load_checkpoint(self.checkpoint_path)
        graph = TaskGraph.from_dict(data["plan"], known_roles=self.roles)
        states = {tid: TaskState(**s) for tid, s in data["states"].items()}
        for s in states.values():
            # Anything in flight at crash time is redone; its partial work was never committed.
            if s.status == "running" or (retry_failed and s.status in ("failed", "skipped")):
                s.status, s.error = "pending", None
        self.budget.restore(data.get("budget", {}))
        tracer = Tracer(self.trace_path, data["run_id"])
        tracer.emit("run_resumed", completed=[t for t, s in states.items() if s.status == "done"])
        gateway = LLMGateway(self.provider, self.budget, tracer)
        blackboard = Blackboard.from_snapshot(data.get("blackboard", {}))
        return await self._execute(data["run_id"], data["goal"], graph, states, blackboard, gateway,
                                   time.monotonic())

    # --------------------------------------------------------------- internals

    def _checkpoint(self, run_id: str, goal: str, graph: TaskGraph, states: dict[str, TaskState],
                    blackboard: Blackboard, tracer: Tracer) -> None:
        save_checkpoint(self.checkpoint_path, {
            "run_id": run_id, "goal": goal, "plan": graph.to_dict(),
            "states": {tid: s.to_dict() for tid, s in states.items()},
            "blackboard": blackboard.snapshot(), "budget": self.budget.to_dict(),
        })
        tracer.emit("checkpoint_saved", done=sum(s.status == "done" for s in states.values()))

    async def _execute(self, run_id: str, goal: str, graph: TaskGraph, states: dict[str, TaskState],
                       blackboard: Blackboard, gateway: LLMGateway, start: float) -> RunResult:
        tracer = gateway.tracer
        worker = Worker(gateway, self.registry, self.workspace, approver=self.approver,
                        require_approval=self.config.require_approval,
                        max_steps=self.config.max_worker_steps, tool_extra=self.tool_extra)
        critic = Critic(gateway, self.workspace)
        running: dict[asyncio.Task[tuple[str, str | None, int, str | None]], str] = {}
        budget_error: BudgetExceeded | None = None
        self._checkpoint(run_id, goal, graph, states, blackboard, tracer)
        try:
            while True:
                if budget_error is None:
                    done = {t for t, s in states.items() if s.status == "done"}
                    not_pending = {t for t, s in states.items() if s.status != "pending"}
                    for tid in graph.ready(done, not_pending):
                        if len(running) >= self.config.max_concurrency:
                            break
                        states[tid].status = "running"
                        tracer.emit("task_started", task=tid, role=graph[tid].role, deps=list(graph[tid].deps))
                        coro = self._run_task(graph, tid, states[tid].attempts, blackboard, worker, critic, tracer)
                        running[asyncio.create_task(coro, name=f"task:{tid}")] = tid
                if not running:
                    break
                finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for fut in finished:
                    tid = running.pop(fut)
                    state = states[tid]
                    exc = None if fut.cancelled() else fut.exception()
                    if exc is not None and not isinstance(exc, Exception):
                        raise exc  # process-level crash: keep the last checkpoint as-is
                    if isinstance(exc, BudgetExceeded):
                        budget_error = exc
                        state.status = "pending"
                        tracer.emit("budget_exceeded", task=tid, error=str(exc))
                        # Cancel only what is still in flight. Futures that already finished in
                        # this same wake-up stay in `running` and are applied normally below:
                        # their work is complete and paid for, so it must not be thrown away.
                        in_flight = [f for f in running if not f.done()]
                        for other in in_flight:
                            other.cancel()
                            otid = running.pop(other)
                            states[otid].status = "pending"
                            tracer.emit("task_cancelled", task=otid, reason="budget")
                        if in_flight:
                            await asyncio.gather(*in_flight, return_exceptions=True)
                    elif exc is not None:
                        self._fail(graph, states, tid, f"{type(exc).__name__}: {exc}", tracer)
                    else:
                        status, output, attempts, error = fut.result()
                        state.attempts = attempts
                        if status == "done":
                            state.status, state.output = "done", output
                            blackboard.put(f"result:{tid}", output, author=tid)
                            tracer.emit("task_finished", task=tid, status="done", attempts=attempts)
                        else:
                            state.output = output
                            self._fail(graph, states, tid, error or "failed", tracer)
                    self._checkpoint(run_id, goal, graph, states, blackboard, tracer)
        finally:
            for fut in running:
                fut.cancel()

        counts = {s: sum(st.status == s for st in states.values()) for s in ("done", "failed", "skipped", "pending")}
        if budget_error is not None:
            status = "budget_exhausted"
        elif counts["done"] == len(states):
            status = "completed"
        else:
            status = "partial" if counts["done"] else "failed"
        elapsed = time.monotonic() - start
        tracer.emit("run_finished", status=status, elapsed_seconds=round(elapsed, 4), usage=self.budget.to_dict(),
                    **counts)
        tracer.close()
        return RunResult(run_id, goal, status, graph, states, self.budget.to_dict(), elapsed, self.trace_path,
                         self.checkpoint_path, None if budget_error is None else str(budget_error), blackboard)

    def _fail(self, graph: TaskGraph, states: dict[str, TaskState], tid: str, error: str, tracer: Tracer) -> None:
        states[tid].status, states[tid].error = "failed", error
        tracer.emit("task_finished", task=tid, status="failed", attempts=states[tid].attempts, error=error)
        for child in sorted(graph.descendants(tid)):
            if states[child].status == "pending":
                states[child].status, states[child].error = "skipped", f"upstream {tid} failed"
                tracer.emit("task_skipped", task=child, because=tid)

    async def _run_task(self, graph: TaskGraph, tid: str, prior_attempts: int, blackboard: Blackboard,
                        worker: Worker, critic: Critic, tracer: Tracer) -> tuple[str, str | None, int, str | None]:
        """Run one task through worker -> critic -> revise. Returns (status, output, attempts, error)."""
        task = graph[tid]
        role = self.roles[task.role]
        deps = {d: str(blackboard.get(f"result:{d}", "")) for d in task.deps}
        feedback: str | None = None
        output: str | None = None
        attempts = prior_attempts
        for _ in range(self.config.max_revisions + 1):
            try:
                result = await worker.run(task, role, attempts, deps, feedback, output)
            except (WorkerFailed, ProviderError) as exc:
                return "failed", output, attempts + 1, str(exc)
            attempts += 1
            output = result.output
            if not self.config.use_critic:
                return "done", output, attempts, None
            verdict = await critic.review(task, output, attempts - 1)
            tracer.emit("critic_verdict", task=tid, attempt=attempts - 1, passed=verdict.passed,
                        source=verdict.source, feedback=verdict.feedback[:300])
            if verdict.passed:
                return "done", output, attempts, None
            feedback = verdict.feedback
        return "failed", output, attempts, f"rejected by critic after {attempts} attempts: {feedback}"
