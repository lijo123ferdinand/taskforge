"""Benchmarks: parallel vs sequential wall time, and task success with vs without the critic.

Everything runs offline on the MockProvider, so read the numbers for what they are:

* **Scheduling** uses a simulated per-call latency (default 50 ms). It measures how
  much wall time the DAG scheduler recovers by overlapping independent tasks,
  i.e. the engine's overhead and parallelism, not any model's speed.
* **Critic** uses seeded fault injection. Each worker attempt faults with
  probability ``--fault-rate``; a share ``--subtle`` of faults are well-formed but
  wrong and pass every check. Success is judged by an independent ground-truth
  grader that recomputes the right answer from the input files, never by the critic.

Usage: python benchmarks/bench.py [--seeds 20] [--latency 0.05] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from taskforge import MockProvider, Orchestrator, RunConfig
from taskforge.tools import safe_calculate

ROOT = Path(__file__).resolve().parent.parent
WORKSPACES = ROOT / "examples" / "workspaces"

DOCS_GOAL = "Research and write a report from the local docs folder"
CSV_GOAL = "Compute and chart stats from the CSV in data/"
CALC_GOALS = [
    "Calculate: (12*7)+3; sqrt(144)*5; 2**10-24",
    "Calculate: 365*24*60; 17**3-4913+1; floor(1000/7); 2**16",
    "Calculate: 3*(4+5)**2; 1e6/125; abs(-42)*3; sqrt(2)*sqrt(8); 99*99",
]


def fresh(name: str, tmp: Path) -> Path:
    dst = tmp / f"{name}_{time.monotonic_ns()}"
    if name == "calc":
        dst.mkdir()
    else:
        shutil.copytree(WORKSPACES / name, dst)
    return dst


def run(provider: MockProvider, ws: Path, goal: str, **cfg):
    return asyncio.run(Orchestrator(provider, ws, config=RunConfig(**cfg)).run(goal))


# ------------------------------------------------------------------ graders


def grade_docs(ws: Path, result) -> bool:
    report = ws / "output" / "report.md"
    if not report.is_file():
        return False
    text = report.read_text()
    for doc in sorted((ws / "docs").glob("*.md")):
        title = next(ln[2:].strip() for ln in doc.read_text().splitlines() if ln.startswith("# "))
        if f"## {title}" not in text or f"_Source: docs/{doc.name}_" not in text:
            return False
    return True


def grade_csv(ws: Path, result) -> bool:
    rows = list(csv.DictReader((ws / "data" / "sales.csv").read_text().splitlines()))
    try:
        stats = json.loads(result.outputs.get("compute_stats", ""))
    except json.JSONDecodeError:
        return False
    report = ws / "output" / "stats_report.md"
    if not report.is_file() or not (ws / "output" / "chart.svg").is_file():
        return False
    text = report.read_text()
    for col in rows[0]:
        try:
            vals = [float(r[col]) for r in rows]
        except ValueError:
            continue
        mean = statistics.fmean(vals)
        if abs(stats.get(col, {}).get("mean", float("nan")) - mean) > 1e-6:
            return False
        if f"| {col} | {mean:.4f} |" not in text:
            return False
    return True


def grade_calc(goal: str, result) -> bool:
    exprs = [e.strip() for e in goal.split(":", 1)[1].split(";")]
    lines = (result.outputs.get("combine") or "").splitlines()
    if len(lines) != len(exprs):
        return False
    for expr, line in zip(exprs, lines):
        m = re.fullmatch(re.escape(expr) + r" = (-?[\d.e+]+)", line)
        if not m or abs(float(m.group(1)) - float(safe_calculate(expr))) > 1e-6:
            return False
    return True


SUITE = [("docs_report", DOCS_GOAL, grade_docs), ("csv_stats", CSV_GOAL, grade_csv)] + [
    ("calc", g, (lambda goal: lambda ws, r: grade_calc(goal, r))(g)) for g in CALC_GOALS
]


# ------------------------------------------------------------------ benches


def bench_scheduling(tmp: Path, latency: float, repeats: int) -> list[dict]:
    rows = []
    for name, goal in (("docs_report", DOCS_GOAL), ("csv_stats", CSV_GOAL), ("calc", CALC_GOALS[2])):
        timings = {}
        for conc in (1, 2, 4, 8):
            samples = []
            for _ in range(repeats):
                ws = fresh(name, tmp)
                r = run(MockProvider(latency=latency), ws, goal, max_concurrency=conc)
                assert r.status == "completed", r.status
                samples.append(r.elapsed_seconds)
                calls = r.usage["calls"]
            timings[conc] = statistics.median(samples)
        rows.append({"dag": name, "tasks": len(r.graph), "levels": len(r.graph.levels()), "llm_calls": calls,
                     "seconds": {str(k): round(v, 3) for k, v in timings.items()},
                     "speedup_4": round(timings[1] / timings[4], 2)})
    return rows


def bench_critic(tmp: Path, seeds: int, fault_rate: float, subtle: float) -> dict:
    out = {}
    for use_critic in (False, True):
        ok = total = calls = tokens = 0
        per_goal: dict[str, list[int]] = {}
        for name, goal, grader in SUITE:
            for seed in range(seeds):
                ws = fresh(name, tmp)
                r = run(MockProvider(fault_rate=fault_rate, subtle_fraction=subtle, seed=seed), ws, goal,
                        use_critic=use_critic, max_revisions=2)
                passed = r.status == "completed" and grader(ws, r)
                ok += passed
                total += 1
                calls += r.usage["calls"]
                tokens += r.usage["input_tokens"] + r.usage["output_tokens"]
                per_goal.setdefault(name if name != "calc" else "calc", []).append(int(passed))
        out["critic" if use_critic else "no_critic"] = {
            "success_rate": round(ok / total, 3), "runs": total,
            "avg_llm_calls": round(calls / total, 1), "avg_tokens": round(tokens / total),
            "per_goal": {k: round(sum(v) / len(v), 3) for k, v in per_goal.items()},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fault-rate", type=float, default=0.2)
    ap.add_argument("--subtle", type=float, default=0.25)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sched = bench_scheduling(tmp, args.latency, args.repeats)
        critic = bench_critic(tmp, args.seeds, args.fault_rate, args.subtle)
    print(f"## Scheduling (simulated {args.latency * 1000:.0f} ms per LLM call, median of {args.repeats})\n")
    print("| DAG | tasks | depth | LLM calls | conc=1 | conc=2 | conc=4 | conc=8 | speedup @4 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sched:
        s = r["seconds"]
        print(f"| {r['dag']} | {r['tasks']} | {r['levels']} | {r['llm_calls']} | {s['1']}s | {s['2']}s | "
              f"{s['4']}s | {s['8']}s | {r['speedup_4']}x |")
    print(f"\n## Critic (fault rate {args.fault_rate}, {args.subtle:.0%} of faults subtle, "
          f"{args.seeds} seeds x {len(SUITE)} goals)\n")
    print("| mode | success | runs | avg LLM calls | avg tokens | docs | csv | calc |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, r in critic.items():
        g = r["per_goal"]
        print(f"| {mode} | {r['success_rate']:.1%} | {r['runs']} | {r['avg_llm_calls']} | {r['avg_tokens']} | "
              f"{g['docs_report']:.0%} | {g['csv_stats']:.0%} | {g['calc']:.0%} |")
    if args.json:
        args.json.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                         "python": sys.version.split()[0],
                                         "scheduling": sched, "critic": critic}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
