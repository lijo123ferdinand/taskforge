"""Crash a run mid-flight, then resume it from the last checkpoint.

The first run "crashes" (a BaseException, like an OOM kill) before its 10th model
call. The resumed run redoes only the tasks that had not completed: no re-plan,
no repeated work, and token spend keeps counting from where it stopped.

    python examples/crash_and_resume.py
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from taskforge import MockProvider, Orchestrator, RunConfig, render_timeline
from taskforge.providers.mock import CrashAfter, SimulatedCrash

HERE = Path(__file__).resolve().parent
GOAL = "Research and write a report from the local docs folder"


def main() -> None:
    ws = Path(tempfile.mkdtemp()) / "docs_report"
    shutil.copytree(HERE / "workspaces" / "docs_report", ws)
    config = RunConfig(max_concurrency=2)

    try:
        asyncio.run(Orchestrator(CrashAfter(MockProvider(), 9), ws, config=config).run(GOAL))
    except SimulatedCrash as exc:
        print(f"crashed: {exc}")
    ckpt = json.loads((ws / ".taskforge" / "checkpoint.json").read_text())
    done = [t for t, s in ckpt["states"].items() if s["status"] == "done"]
    print(f"checkpoint has {len(done)}/{len(ckpt['states'])} tasks done: {done}")

    provider = MockProvider()
    result = asyncio.run(Orchestrator(provider, ws, config=config).resume())
    print(f"resumed run: status={result.status}, model calls after resume={provider.calls}\n")
    print(render_timeline(result.trace_path))


if __name__ == "__main__":
    main()
