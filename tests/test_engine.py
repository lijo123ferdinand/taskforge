import asyncio
import json
import time

import pytest
from conftest import CALC_GOAL, CSV_GOAL, DOCS_GOAL

from taskforge import (Budget, MockProvider, Orchestrator, PolicyApprover, RunConfig, deny_all,
                       render_timeline)
from taskforge.checkpoint import load_checkpoint
from taskforge.providers.base import Completion, ProviderError, ToolCall, Usage
from taskforge.providers.mock import CrashAfter, SimulatedCrash
from taskforge.trace import load_trace


def run(provider, ws, goal, **kw):
    budget = kw.pop("budget", None)
    approver = kw.pop("approver", None)
    extra = {"approver": approver} if approver else {}
    orch = Orchestrator(provider, ws, config=RunConfig(**kw), budget=budget, **extra)
    return asyncio.run(orch.run(goal)), orch


def events(result, kind):
    return [e for e in load_trace(result.trace_path) if e["type"] == kind]


def test_docs_report_end_to_end(docs_ws):
    result, _ = run(MockProvider(), docs_ws, DOCS_GOAL)
    assert result.status == "completed"
    assert len(result.graph) == 8  # 6 summaries + outline + write_report
    report = (docs_ws / "output/report.md").read_text()
    for title in ("Caching Strategies", "Message Queues", "Retries and Timeouts"):
        assert f"## {title}" in report
    assert "_Source: docs/queues.md_" in report
    assert result.blackboard.get("result:outline").startswith("## ")


def test_csv_stats_numbers_are_exact(csv_ws):
    import csv
    import statistics
    result, _ = run(MockProvider(), csv_ws, CSV_GOAL)
    assert result.status == "completed"
    stats = json.loads(result.outputs["compute_stats"])
    rows = list(csv.DictReader((csv_ws / "data/sales.csv").read_text().splitlines()))
    assert stats["units"]["mean"] == pytest.approx(statistics.fmean(float(r["units"]) for r in rows))
    assert stats["region"] == {"type": "categorical", "distinct": 4}
    assert (csv_ws / "output/chart.svg").read_text().startswith("<svg")
    assert "| units |" in (csv_ws / "output/stats_report.md").read_text()


def test_calc_goal(empty_ws):
    result, _ = run(MockProvider(), empty_ws, CALC_GOAL)
    assert result.status == "completed"
    assert result.outputs["combine"].splitlines() == ["(12*7)+3 = 87", "sqrt(144)*5 = 60", "2**10-24 = 1000"]


def test_independent_tasks_run_concurrently_and_bounded(docs_ws):
    result, _ = run(MockProvider(latency=0.02), docs_ws, DOCS_GOAL, max_concurrency=3)
    live, peak = 0, 0
    for e in load_trace(result.trace_path):
        if e["type"] == "task_started":
            live += 1
            peak = max(peak, live)
        elif e["type"] == "task_finished":
            live -= 1
    assert peak == 3  # six independent summaries, capped at three


def test_parallel_is_faster_than_sequential(docs_ws, tmp_path):
    import shutil
    other = shutil.copytree(docs_ws, tmp_path / "seq")
    t0 = time.monotonic()
    run(MockProvider(latency=0.03), docs_ws, DOCS_GOAL, max_concurrency=6)
    parallel = time.monotonic() - t0
    t0 = time.monotonic()
    run(MockProvider(latency=0.03), other, DOCS_GOAL, max_concurrency=1)
    sequential = time.monotonic() - t0
    assert parallel < sequential * 0.7


def test_critic_rejects_and_worker_revises(docs_ws):
    result, _ = run(MockProvider(fault_rate=0.4, subtle_fraction=0.0, seed=0), docs_ws, DOCS_GOAL)
    rejected = [e for e in events(result, "critic_verdict") if not e["passed"]]
    assert rejected and result.status == "completed"
    assert any(s.attempts > 1 for s in result.states.values())


def test_without_critic_faulty_output_is_accepted(docs_ws):
    result, _ = run(MockProvider(fault_rate=0.4, subtle_fraction=0.0, seed=0), docs_ws, DOCS_GOAL, use_critic=False)
    assert result.status == "completed"
    assert not events(result, "critic_verdict")
    summaries = [o for t, o in result.outputs.items() if t.startswith("summarize_")]
    assert any("Source:" not in s for s in summaries)  # an injected fault slipped through


def test_permanent_failure_skips_dependents_only(empty_ws):
    result, _ = run(MockProvider(fault_rate=1.0, subtle_fraction=0.0), empty_ws, CALC_GOAL, max_revisions=1)
    assert {s.status for t, s in result.states.items() if t.startswith("calc_")} == {"failed"}
    assert result.states["combine"].status == "skipped"
    assert result.states["calc_1"].attempts == 2
    assert result.status == "failed"


def test_budget_hard_stop_then_resume_with_more_budget(docs_ws):
    result, orch = run(MockProvider(), docs_ws, DOCS_GOAL, budget=Budget(max_tokens=4000), max_concurrency=2)
    assert result.status == "budget_exhausted"
    assert result.usage["input_tokens"] + result.usage["output_tokens"] <= 4000 + 1500  # bounded overshoot
    done_before = {t for t, s in result.states.items() if s.status == "done"}
    assert done_before and len(done_before) < len(result.graph)
    assert all(s.status in ("done", "pending") for s in result.states.values())

    orch2 = Orchestrator(MockProvider(), docs_ws, budget=Budget(max_tokens=100_000))
    resumed = asyncio.run(orch2.resume())
    assert resumed.status == "completed"
    restarted = {e["task"] for e in load_trace(resumed.trace_path)
                 if e["type"] == "task_started" and e["t"] > events(result, "run_finished")[0]["t"]}
    assert restarted.isdisjoint(done_before)
    assert resumed.usage["input_tokens"] > result.usage["input_tokens"]  # spend carried over


def test_crash_then_resume_from_last_completed_node(docs_ws):
    with pytest.raises(SimulatedCrash):
        run(CrashAfter(MockProvider(), 9), docs_ws, DOCS_GOAL, max_concurrency=2)
    ckpt = load_checkpoint(docs_ws / ".taskforge/checkpoint.json")
    done = {t for t, s in ckpt["states"].items() if s["status"] == "done"}
    assert 0 < len(done) < 8

    fresh = MockProvider()
    resumed = asyncio.run(Orchestrator(fresh, docs_ws).resume())
    assert resumed.status == "completed"
    assert (docs_ws / "output/report.md").exists()
    after = [e for e in load_trace(resumed.trace_path)]
    resume_idx = next(i for i, e in enumerate(after) if e["type"] == "run_resumed")
    restarted = {e["task"] for e in after[resume_idx:] if e["type"] == "task_started"}
    assert restarted == set(ckpt["states"]) - done
    assert not any(e["type"] == "plan_created" for e in after[resume_idx:])  # no re-plan
    assert "(planner)" in render_timeline(resumed.trace_path)


def test_approval_denied_blocks_the_write(docs_ws):
    result, _ = run(MockProvider(), docs_ws, DOCS_GOAL, approver=deny_all,
                    require_approval={"write_file"}, max_revisions=1)
    assert result.states["write_report"].status == "failed"
    assert not (docs_ws / "output/report.md").exists()
    assert result.status == "partial"
    decisions = events(result, "approval_decision")
    assert decisions and not any(d["approved"] for d in decisions)


def test_policy_approver_allows_matching_calls(docs_ws):
    approver = PolicyApprover(lambda r: r.arguments["path"].startswith("output/"))
    result, _ = run(MockProvider(), docs_ws, DOCS_GOAL, approver=approver, require_approval={"write_file"})
    assert result.status == "completed"
    assert events(result, "approval_requested")[0]["tool"] == "write_file"


class Scripted:
    """Provider driven by per-agent lists of canned completions."""
    name = "scripted"

    def __init__(self, **by_agent):
        self.by_agent = {k: list(v) for k, v in by_agent.items()}

    async def complete(self, request):
        item = self.by_agent[request.agent].pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Completion):
            return item
        return Completion(text=item, usage=Usage(10, 5))


PLAN_ONE = json.dumps({"tasks": [{"id": "only", "description": "say hi", "role": "writer"}]})


def test_planner_gets_validator_feedback_and_replans(empty_ws):
    cyclic = json.dumps({"tasks": [{"id": "a", "description": "x", "deps": ["b"]},
                                   {"id": "b", "description": "y", "deps": ["a"]}]})
    provider = Scripted(planner=[cyclic, "```json\n" + PLAN_ONE + "\n```"], worker=["hi"])
    result, _ = run(provider, empty_ws, "anything")
    assert result.status == "completed" and result.outputs["only"] == "hi"
    assert "dependency cycle" in events(result, "plan_rejected")[0]["error"]


def test_disallowed_tool_is_refused_and_reported_back(empty_ws):
    sneaky = Completion("", [ToolCall("c1", "python_eval", {"code": "result = 1"})], Usage(1, 1))
    provider = Scripted(planner=[PLAN_ONE], worker=[sneaky, "done without it"])
    result, _ = run(provider, empty_ws, "anything")
    assert events(result, "tool_denied")[0]["tool"] == "python_eval"  # writers never get python_eval
    assert result.outputs["only"] == "done without it"


def test_transient_provider_errors_are_retried(empty_ws, monkeypatch):
    provider = Scripted(planner=[ProviderError("429", retryable=True), PLAN_ONE], worker=["ok"])
    orch = Orchestrator(provider, empty_ws)
    monkeypatch.setattr("taskforge.agents.LLMGateway.__init__.__defaults__", (2, 0.0))
    result = asyncio.run(orch.run("x"))
    assert result.status == "completed"
    assert events(result, "llm_retry")


def test_worker_step_limit_fails_task(empty_ws):
    loop = [Completion("", [ToolCall(f"c{i}", "read_file", {"path": "nope"})], Usage(1, 1)) for i in range(3)]
    provider = Scripted(planner=[PLAN_ONE], worker=loop)
    result, _ = run(provider, empty_ws, "x", max_worker_steps=3)
    assert result.states["only"].status == "failed"
    assert "no final answer after 3 steps" in result.states["only"].error


def test_timeline_renders_every_task(csv_ws):
    result, _ = run(MockProvider(), csv_ws, CSV_GOAL)
    text = render_timeline(result.trace_path)
    for tid in result.graph.tasks:
        assert tid in text
    assert "run completed" in text and "|" in text
