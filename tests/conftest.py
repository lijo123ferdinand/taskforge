import shutil
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workspaces"

DOCS_GOAL = "Research and write a report from the local docs folder"
CSV_GOAL = "Compute and chart stats from the CSV in data/"
CALC_GOAL = "Calculate: (12*7)+3; sqrt(144)*5; 2**10-24"


@pytest.fixture
def docs_ws(tmp_path: Path) -> Path:
    return Path(shutil.copytree(EXAMPLES / "docs_report", tmp_path / "docs_report"))


@pytest.fixture
def csv_ws(tmp_path: Path) -> Path:
    return Path(shutil.copytree(EXAMPLES / "csv_stats", tmp_path / "csv_stats"))


@pytest.fixture
def empty_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws
