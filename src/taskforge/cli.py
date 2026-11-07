"""Command line: ``taskforge run | resume | plan | timeline``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .agents import LLMGateway, Planner
from .approval import ConsoleApprover, auto_approve, deny_all
from .budget import Budget
from .engine import Orchestrator, RunConfig, RunResult
from .providers import make_provider
from .roles import DEFAULT_ROLES
from .timeline import render_timeline
from .tools.builtin import default_registry, sandbox_manifest
from .trace import Tracer


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", "-w", type=Path, default=Path("."), help="sandbox root (default: .)")
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "ollama"])
    p.add_argument("--model", default=None, help="model id for real providers")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-critic", action="store_true", help="accept worker output without review")
    p.add_argument("--max-revisions", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=None, help="hard token budget for the run")
    p.add_argument("--max-cost", type=float, default=None, help="hard USD budget (needs --price-in/--price-out)")
    p.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    p.add_argument("--approve", default="", help="comma-separated tools that need human approval")
    p.add_argument("--yes", action="store_true", help="auto-approve gated tools")
    p.add_argument("--deny", action="store_true", help="auto-deny gated tools")
    p.add_argument("--allow-network", action="store_true", help="register the web_fetch tool")
    p.add_argument("--latency", type=float, default=0.0, help="mock only: simulated seconds per call")
    p.add_argument("--fault-rate", type=float, default=0.0, help="mock only: injected fault probability")
    p.add_argument("--seed", type=int, default=0, help="mock only: fault RNG seed")
    p.add_argument("--quiet", "-q", action="store_true")


def _build(args: argparse.Namespace) -> Orchestrator:
    mock_kwargs = {"latency": args.latency, "fault_rate": args.fault_rate, "seed": args.seed}
    provider = make_provider(args.provider, args.model, **(mock_kwargs if args.provider == "mock" else {}))
    gated = frozenset(t.strip() for t in args.approve.split(",") if t.strip())
    approver = auto_approve if args.yes else deny_all if args.deny else ConsoleApprover()
    config = RunConfig(max_concurrency=args.concurrency, use_critic=not args.no_critic,
                       max_revisions=args.max_revisions, require_approval=gated)
    budget = Budget(max_tokens=args.max_tokens, max_cost_usd=args.max_cost,
                    usd_per_mtok_in=args.price_in, usd_per_mtok_out=args.price_out)
    return Orchestrator(provider, args.workspace, config=config, registry=default_registry(
        allow_network=args.allow_network), approver=approver, budget=budget)


def _report(result: RunResult, quiet: bool) -> int:
    if not quiet:
        print(render_timeline(result.trace_path))
        print()
        for tid, state in result.states.items():
            head = (state.output or state.error or "").strip().splitlines()
            print(f"[{state.status:>7}] {tid}: {head[0][:100] if head else ''}")
        print()
    print(json.dumps({"run_id": result.run_id, "status": result.status, "usage": result.usage,
                      "elapsed_seconds": round(result.elapsed_seconds, 3), "trace": str(result.trace_path),
                      "checkpoint": str(result.checkpoint_path), "error": result.error}, indent=2))
    return 0 if result.status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taskforge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="plan and execute a goal")
    run.add_argument("goal")
    _common(run)
    resume = sub.add_parser("resume", help="resume the run checkpointed in the workspace")
    resume.add_argument("--retry-failed", action="store_true")
    _common(resume)
    plan = sub.add_parser("plan", help="only plan: print the validated DAG")
    plan.add_argument("goal")
    _common(plan)
    tl = sub.add_parser("timeline", help="render a trace.jsonl as a terminal timeline")
    tl.add_argument("trace", type=Path)
    tl.add_argument("--width", type=int, default=48)
    args = parser.parse_args(argv)

    if args.cmd == "timeline":
        print(render_timeline(args.trace, width=args.width))
        return 0
    if args.cmd == "plan":
        orch = _build(args)
        gateway = LLMGateway(orch.provider, orch.budget, Tracer())
        graph = asyncio.run(Planner(gateway, DEFAULT_ROLES, orch.registry).plan(
            args.goal, sandbox_manifest(orch.workspace)))
        for level, ids in enumerate(graph.levels()):
            for tid in ids:
                t = graph[tid]
                print(f"L{level} {tid:<24} role={t.role:<10} deps={list(t.deps)} tools={list(t.tools or [])}")
        return 0
    orch = _build(args)
    try:
        result = asyncio.run(orch.run(args.goal) if args.cmd == "run" else orch.resume(retry_failed=args.retry_failed))
    except KeyboardInterrupt:
        print("interrupted; resume with `taskforge resume -w <workspace>`", file=sys.stderr)
        return 130
    return _report(result, args.quiet)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
