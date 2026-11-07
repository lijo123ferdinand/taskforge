"""Render a JSONL trace as a terminal Gantt chart.

ASCII only, on purpose: the output must survive Windows consoles (cp1252), CI
logs and copy-paste into issues. Example::

    task               0.00s                                   1.23s
    summarize_alpha    |#########.................................|  done  1 att  1 tools
    write_report       |..........................##############..|  done  2 att  1 tools
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .trace import load_trace

__all__ = ["render_timeline", "summarize_trace"]


def summarize_trace(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    events = list(events)
    spans: dict[str, dict[str, Any]] = {}
    tools = defaultdict(int)
    tokens = defaultdict(int)
    rejections = defaultdict(int)
    plan_start = 0.0
    for e in events:
        tid = e.get("task")
        kind = e["type"]
        if kind == "run_started" or kind == "run_resumed":
            plan_start = e["t"]
        elif kind == "plan_created":
            # Planning is on the critical path of every run; show it as its own bar.
            spans["(planner)"] = {"start": plan_start, "end": e["t"], "status": "done", "attempts": e.get("attempt", 0) + 1}
        elif kind == "task_started":
            # Keep the latest start: after a crash+resume the redo is what finished.
            spans[tid] = {"start": e["t"], "end": None, "status": "running", "attempts": 0}
        elif kind in ("task_finished", "task_cancelled") and tid in spans:
            spans[tid]["end"] = e["t"]
            spans[tid]["status"] = e.get("status", "cancelled")
            spans[tid]["attempts"] = e.get("attempts", 0)
        elif kind == "task_skipped":
            spans.setdefault(tid, {"start": e["t"], "end": e["t"], "status": "skipped", "attempts": 0})
        elif kind == "tool_call" and tid:
            tools[tid] += 1
        elif kind == "llm_call" and e.get("agent") == "planner":
            tokens["(planner)"] += e.get("input_tokens", 0) + e.get("output_tokens", 0)
        elif kind == "llm_call" and tid:
            tokens[tid] += e.get("input_tokens", 0) + e.get("output_tokens", 0)
        elif kind == "critic_verdict" and not e.get("passed"):
            rejections[tid] += 1
    finished = next((e for e in reversed(events) if e["type"] == "run_finished"), None)
    return {"spans": spans, "tools": dict(tools), "tokens": dict(tokens), "rejections": dict(rejections),
            "finished": finished, "end": max((e["t"] for e in events), default=0.0)}


def render_timeline(trace: Path | str | list[dict[str, Any]], width: int = 48) -> str:
    events = load_trace(trace) if isinstance(trace, (str, Path)) else list(trace)
    if not events:
        return "(empty trace)"
    info = summarize_trace(events)
    spans = info["spans"]
    t_end = info["end"] or 1e-9
    name_w = max([len("task")] + [len(t) for t in spans])
    lines = [f"{'task'.ljust(name_w)}  0.00s{' ' * (width - 9)}{t_end:.2f}s"]
    for tid, span in sorted(spans.items(), key=lambda kv: kv[1]["start"]):
        end = span["end"] if span["end"] is not None else t_end
        a = int(span["start"] / t_end * width)
        b = max(a + 1, int(round(end / t_end * width)))
        fill = "-" if span["status"] == "skipped" else ("x" if span["status"] == "failed" else "#")
        bar = "." * a + fill * (b - a) + "." * (width - b)
        extra = f"{span['status']:<9}{span['attempts']} att {info['tools'].get(tid, 0):>2} tools " \
                f"{info['tokens'].get(tid, 0):>6} tok"
        if info["rejections"].get(tid):
            extra += f"  critic rejected x{info['rejections'][tid]}"
        lines.append(f"{tid.ljust(name_w)}  |{bar[:width]}|  {extra}")
    fin = info["finished"]
    if fin:
        usage = fin.get("usage", {})
        lines.append(
            f"run {fin.get('status')}: {fin.get('done', 0)} done, {fin.get('failed', 0)} failed, "
            f"{fin.get('skipped', 0)} skipped in {fin.get('elapsed_seconds', 0):.2f}s; "
            f"{usage.get('calls', 0)} LLM calls, {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens")
    return "\n".join(lines)
