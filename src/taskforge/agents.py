"""The three agents (planner, worker, critic) and the LLM gateway they share.

Agents are plain classes over a :class:`Provider`; there is no agent framework.
Each one is a short, readable loop, because when a multi-agent run goes wrong
you need to be able to read *exactly* what each agent was told.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .approval import ApprovalRequest, Approver, auto_approve
from .budget import Budget
from .checks import run_checks
from .dag import PlanError, Task, TaskGraph
from .providers.base import (Completion, CompletionRequest, Message, Provider, ProviderError, request_text,
                             estimate_tokens)
from .roles import Role, effective_tools
from .tools.base import ToolContext, ToolRegistry
from .trace import Tracer

__all__ = ["LLMGateway", "Planner", "Worker", "Critic", "WorkerFailed", "Verdict", "WorkerResult",
           "extract_json", "format_worker_prompt"]


class WorkerFailed(RuntimeError):
    """The worker could not produce an output (step limit, provider error...)."""


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of model text.

    Models wrap JSON in ```json fences or add a sentence before it no matter how
    firmly they are told not to; being lenient here is cheaper than a retry.
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(candidate):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(candidate[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in model output")


# ------------------------------------------------------------------ gateway


class LLMGateway:
    """Every model call goes through here: budget gate, retries, usage accounting, tracing."""

    def __init__(self, provider: Provider, budget: Budget, tracer: Tracer, max_retries: int = 2,
                 backoff_seconds: float = 0.5) -> None:
        self.provider = provider
        self.budget = budget
        self.tracer = tracer
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    async def complete(self, request: CompletionRequest, task_id: str = "") -> Completion:
        self.budget.check_before_call(estimate_tokens(request_text(request)))
        attempt = 0
        while True:
            start = time.monotonic()
            try:
                completion = await self.provider.complete(request)
                break
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    self.tracer.emit("llm_error", agent=request.agent, task=task_id, error=str(exc))
                    raise
                # Exponential backoff; transient 429/5xx usually clear within seconds.
                delay = self.backoff_seconds * (2 ** attempt)
                self.tracer.emit("llm_retry", agent=request.agent, task=task_id, error=str(exc), delay=delay)
                attempt += 1
                await asyncio.sleep(delay)
        self.tracer.emit(
            "llm_call", agent=request.agent, task=task_id, model=completion.model,
            input_tokens=completion.usage.input_tokens, output_tokens=completion.usage.output_tokens,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            tool_calls=[c.name for c in completion.tool_calls],
        )
        # Charging may raise BudgetExceeded; the trace above already records the call.
        self.budget.charge(completion.usage, request.agent)
        return completion


# ------------------------------------------------------------------ planner

PLANNER_SYSTEM = """You are the planner of a multi-agent system. Decompose the GOAL into a small DAG of tasks.
Reply with ONLY a JSON object:
{"tasks": [{"id": "snake_case_id", "description": "what to do, concretely", "role": "<one of ROLES>",
            "deps": ["ids this task needs"], "tools": ["subset of the role's tools, optional"],
            "acceptance": ["natural-language criteria a reviewer can verify"],
            "checks": [{"type": "contains", "value": "..."} | {"type": "regex", "pattern": "..."} |
                       {"type": "min_length", "value": 50} | {"type": "json_keys", "keys": ["..."]} |
                       {"type": "file_exists", "path": "..."} | {"type": "not_contains", "value": "..."}]}]}
Rules: no cycles; deps must reference earlier ids; maximise independent tasks so they can run in parallel;
keep it under 12 tasks; refer only to files listed in WORKSPACE FILES."""


class Planner:
    def __init__(self, gateway: LLMGateway, roles: dict[str, Role], registry: ToolRegistry,
                 max_attempts: int = 3) -> None:
        self.gateway = gateway
        self.roles = roles
        self.registry = registry
        self.max_attempts = max_attempts

    def _prompt(self, goal: str, manifest: list[str]) -> str:
        files = "\n".join(f"- {line}" for line in manifest) or "- (empty)"
        roles = "\n".join(
            f"- {r.name}: tools {effective_tools(r, None, self.registry.names())}" for r in self.roles.values())
        return f"GOAL:\n{goal}\n\nWORKSPACE FILES:\n{files}\n\nROLES:\n{roles}\n"

    async def plan(self, goal: str, manifest: list[str]) -> TaskGraph:
        messages = [Message("user", self._prompt(goal, manifest))]
        last_error = ""
        for attempt in range(self.max_attempts):
            completion = await self.gateway.complete(
                CompletionRequest(PLANNER_SYSTEM, list(messages), max_tokens=2048, agent="planner"), "planner")
            try:
                graph = TaskGraph.from_dict(extract_json(completion.text), known_roles=self.roles)
                self.gateway.tracer.emit("plan_created", attempt=attempt, plan=graph.to_dict(),
                                         levels=graph.levels())
                return graph
            except (ValueError, PlanError) as exc:
                # Feed the validator's precise complaint back; models fix "cycle a -> b -> a"
                # far more reliably than "invalid plan".
                last_error = str(exc)
                self.gateway.tracer.emit("plan_rejected", attempt=attempt, error=last_error)
                messages += [Message("assistant", completion.text),
                             Message("user", f"That plan is invalid: {last_error}. Reply with a corrected JSON plan.")]
        raise PlanError(f"planner failed after {self.max_attempts} attempts: {last_error}")


# ------------------------------------------------------------------ worker


@dataclass
class WorkerResult:
    output: str
    steps: int
    tool_calls: int


def format_worker_prompt(task: Task, attempt: int, dep_outputs: dict[str, str],
                         feedback: str | None, previous_output: str | None) -> str:
    spec = {"id": task.id, "attempt": attempt, "deps": list(task.deps)}
    lines = [f"TASK_SPEC: {json.dumps(spec)}", "", f"TASK: {task.description}"]
    if task.acceptance or task.checks:
        lines.append("ACCEPTANCE CRITERIA:")
        lines += [f"- {a}" for a in task.acceptance]
        lines += [f"- [check] {json.dumps(c)}" for c in task.checks]
    if dep_outputs:
        lines.append("DEPENDENCY OUTPUTS:")
        for dep, out in dep_outputs.items():
            lines += [f"<<<dep:{dep}>>>", out, f"<<<end:{dep}>>>"]
    if feedback:
        lines += ["", f"REVISION FEEDBACK (attempt {attempt}): {feedback}"]
        if previous_output is not None:
            lines += ["PREVIOUS OUTPUT:", "<<<previous>>>", previous_output, "<<<end:previous>>>"]
    return "\n".join(lines)


class Worker:
    def __init__(self, gateway: LLMGateway, registry: ToolRegistry, sandbox: Path, *,
                 approver: Approver = auto_approve, require_approval: frozenset[str] = frozenset(),
                 max_steps: int = 8, tool_extra: dict[str, Any] | None = None) -> None:
        self.gateway = gateway
        self.registry = registry
        self.sandbox = sandbox
        self.approver = approver
        self.require_approval = require_approval
        self.max_steps = max_steps
        self.tool_extra = tool_extra or {}

    async def run(self, task: Task, role: Role, attempt: int, dep_outputs: dict[str, str],
                  feedback: str | None = None, previous_output: str | None = None) -> WorkerResult:
        tracer = self.gateway.tracer
        allowed = effective_tools(role, task.tools, self.registry.names())
        specs = self.registry.specs(allowed)
        messages = [Message("user", format_worker_prompt(task, attempt, dep_outputs, feedback, previous_output))]
        ctx = ToolContext(sandbox=self.sandbox, task_id=task.id, extra=self.tool_extra)
        n_tools = 0
        for step in range(1, self.max_steps + 1):
            completion = await self.gateway.complete(
                CompletionRequest(role.system_prompt, list(messages), specs, max_tokens=4096, agent="worker"),
                task.id)
            if not completion.tool_calls:
                return WorkerResult(completion.text.strip(), step, n_tools)
            messages.append(Message("assistant", completion.text, tool_calls=list(completion.tool_calls)))
            # Sequential within a task: a model that emits write_file then read_file in one
            # turn expects them to happen in that order. Parallelism lives *between* tasks.
            for call in completion.tool_calls:
                n_tools += 1
                ok, content = await self._invoke(task, call.name, call.arguments, allowed, ctx)
                messages.append(Message("tool", content, tool_call_id=call.id, name=call.name, is_error=not ok))
        raise WorkerFailed(f"task {task.id}: no final answer after {self.max_steps} steps")

    async def _invoke(self, task: Task, name: str, args: dict[str, Any], allowed: list[str],
                      ctx: ToolContext) -> tuple[bool, str]:
        tracer = self.gateway.tracer
        preview = json.dumps(args, default=str)[:300]
        if name not in allowed:
            tracer.emit("tool_denied", task=task.id, tool=name, reason="not in allow-list")
            return False, f"ERROR: tool {name!r} is not allowed for this task; allowed: {allowed}"
        if name in self.require_approval:
            tracer.emit("approval_requested", task=task.id, tool=name, args=preview)
            approved = await self.approver(ApprovalRequest(task.id, name, args))
            tracer.emit("approval_decision", task=task.id, tool=name, approved=approved)
            if not approved:
                return False, f"ERROR: {name} call was denied by the human approver"
        start = time.monotonic()
        result = await self.registry.invoke(name, args, ctx)
        tracer.emit("tool_call", task=task.id, tool=name, args=preview, ok=result.ok,
                    ms=round((time.monotonic() - start) * 1000, 2), result=result.content[:200])
        return result.ok, result.to_text()


# ------------------------------------------------------------------ critic

CRITIC_SYSTEM = """You are a strict reviewer. Decide whether the OUTPUT satisfies the task's acceptance criteria.
Reply with ONLY JSON: {"pass": true|false, "feedback": "if failing: the specific, actionable fix"}"""


@dataclass
class Verdict:
    passed: bool
    feedback: str
    source: str  # "checks" | "llm" | "none"
    details: list[dict[str, Any]] = field(default_factory=list)


class Critic:
    def __init__(self, gateway: LLMGateway, sandbox: Path) -> None:
        self.gateway = gateway
        self.sandbox = sandbox

    async def review(self, task: Task, output: str, attempt: int) -> Verdict:
        results = run_checks(task.checks, output, self.sandbox)
        failed = [msg for _, ok, msg in results if not ok]
        details = [{"check": c, "ok": ok} for c, ok, _ in results]
        if failed:
            # Mechanical failure: no need to pay for an LLM opinion.
            return Verdict(False, "; ".join(failed), "checks", details)
        if not task.acceptance:
            return Verdict(True, "", "checks" if task.checks else "none", details)
        prompt = (f"TASK_SPEC: {json.dumps({'id': task.id, 'attempt': attempt})}\n"
                  f"TASK: {task.description}\nACCEPTANCE CRITERIA:\n"
                  + "\n".join(f"- {a}" for a in task.acceptance)
                  + f"\nOUTPUT:\n<<<output>>>\n{output}\n<<<end:output>>>")
        completion = await self.gateway.complete(
            CompletionRequest(CRITIC_SYSTEM, [Message("user", prompt)], max_tokens=512, agent="critic"), task.id)
        try:
            data = extract_json(completion.text)
            passed = bool(data["pass"])
            return Verdict(passed, str(data.get("feedback", "")), "llm", details)
        except (ValueError, KeyError, TypeError):
            # Fail-open on an unreadable verdict: deterministic checks already passed, and
            # failing closed would burn the revision budget on the critic's formatting bug.
            self.gateway.tracer.emit("critic_unparsable", task=task.id, text=completion.text[:200])
            return Verdict(True, "critic verdict unparsable; accepted on checks", "llm", details)
