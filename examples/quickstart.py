"""Library quickstart: run the docs-report goal on a scratch copy of the bundled workspace.

    python examples/quickstart.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from taskforge import Budget, MockProvider, Orchestrator, PolicyApprover, RunConfig, render_timeline

HERE = Path(__file__).resolve().parent


async def main() -> None:
    ws = Path(tempfile.mkdtemp()) / "docs_report"
    shutil.copytree(HERE / "workspaces" / "docs_report", ws)

    orch = Orchestrator(
        MockProvider(latency=0.05),            # swap for make_provider("anthropic") with a key
        ws,
        config=RunConfig(max_concurrency=4, require_approval={"write_file"}),
        # Human-in-the-loop gate, here as a policy: writes are only allowed under output/.
        approver=PolicyApprover(lambda req: req.arguments["path"].startswith("output/")),
        budget=Budget(max_tokens=50_000),
    )
    result = await orch.run("Research and write a report from the local docs folder")
    print(render_timeline(result.trace_path))
    print(f"\nstatus={result.status}  report={ws / 'output' / 'report.md'}")
    print((ws / "output" / "report.md").read_text()[:400])


if __name__ == "__main__":
    asyncio.run(main())
