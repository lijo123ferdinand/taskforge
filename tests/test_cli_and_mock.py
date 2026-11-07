import asyncio
import json

from conftest import CALC_GOAL, CSV_GOAL, DOCS_GOAL

from taskforge.cli import main
from taskforge.dag import TaskGraph
from taskforge.providers.base import CompletionRequest, Message
from taskforge.providers.mock import MockProvider
from taskforge.roles import DEFAULT_ROLES
from taskforge.tools.builtin import sandbox_manifest


def plan_for(goal, ws):
    files = "\n".join(f"- {line}" for line in sandbox_manifest(ws))
    req = CompletionRequest("", [Message("user", f"GOAL:\n{goal}\n\nWORKSPACE FILES:\n{files}\n\nROLES:\n")],
                            agent="planner")
    from taskforge.agents import extract_json
    return TaskGraph.from_dict(extract_json(asyncio.run(MockProvider().complete(req)).text), DEFAULT_ROLES)


def test_mock_plans_are_valid_and_parallel(docs_ws, csv_ws, empty_ws):
    docs = plan_for(DOCS_GOAL, docs_ws)
    assert len(docs.levels()[0]) == 6
    csvp = plan_for(CSV_GOAL, csv_ws)
    assert csvp.levels() == [["profile_data"], ["compute_stats", "chart"], ["write_summary"]]
    assert csvp["compute_stats"].checks[0]["keys"] == ["region", "units", "unit_price", "discount_pct"]
    assert len(plan_for(CALC_GOAL, empty_ws).levels()[0]) == 3
    assert list(plan_for("Write a poem", empty_ws).tasks) == ["respond"]


def test_fault_injection_is_deterministic():
    a = [MockProvider(fault_rate=0.5, seed=3)._fault(f"t{i}", 0, True) for i in range(50)]
    b = [MockProvider(fault_rate=0.5, seed=3)._fault(f"t{i}", 0, True) for i in range(50)]
    assert a == b and {"obvious", None} <= set(a)


def test_cli_run_plan_timeline(docs_ws, capsys):
    assert main(["run", DOCS_GOAL, "-w", str(docs_ws), "-q"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert main(["timeline", summary["trace"]]) == 0
    assert "write_report" in capsys.readouterr().out
    assert main(["plan", DOCS_GOAL, "-w", str(docs_ws)]) == 0
    assert "L0 summarize_caching" in capsys.readouterr().out


def test_cli_budget_exit_code_and_resume(docs_ws, capsys):
    assert main(["run", DOCS_GOAL, "-w", str(docs_ws), "-q", "--max-tokens", "3000"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "budget_exhausted"
    assert main(["resume", "-w", str(docs_ws), "-q", "--max-tokens", "100000"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
