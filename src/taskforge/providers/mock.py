"""A deterministic, offline provider that behaves like a competent model on the bundled goals.

What it is: a *scripted policy*. It reads the same prompts a real model would get
(the planner's GOAL + WORKSPACE FILES, a worker's TASK_SPEC + tool results, the
critic's OUTPUT) and answers with real plans, real tool calls and outputs computed
from real tool results. It does not fake the engine: every plan is validated,
every tool call goes through schema validation and the sandbox, every check runs.

What it is not: a language model. It recognises three goal families (report from
docs, CSV stats + chart, arithmetic) and the task ids its own planner emits. For
anything else it returns a one-task plan whose output says so.

Fault injection (``fault_rate``) makes it imperfect on purpose, so the critic has
something to catch. Each worker attempt independently, with seeded probability
``fault_rate``, produces either an *obvious* fault (missing citation, file never
written, non-numeric answer) that deterministic checks can catch, or, with share
``subtle_fraction``, a *subtle* one (wrong title, mean off by one, off-by-one
arithmetic) that passes the checks and only a ground-truth grader detects. That
split is what keeps the critic benchmark honest: it cannot reach 100%.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from .base import Completion, CompletionRequest, Message, ToolCall, Usage, estimate_tokens, request_text

__all__ = ["MockProvider", "CrashAfter", "SimulatedCrash"]

_STOPWORDS = set("""a an the and or but of to in on for with by from as at is are was were be been being this that
these those it its into than then so such not no can will would should could may might must also more most very
which who whom what when where why how all any each both few other some our your their they them we you he she
his her has have had do does did over under between about after before during through per via if""".split())

_MANIFEST_RE = re.compile(r"^- (\S+) \((\d+) bytes\)(?: header: (.*))?$")
_DEP_RE = re.compile(r"<<<dep:(\S+?)>>>\n(.*?)\n<<<end:\1>>>", re.DOTALL)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "item"


class SimulatedCrash(BaseException):
    """Stands in for a process crash (OOM kill, Ctrl-C). Deliberately *not* an Exception
    subclass, so no ``except Exception`` handler in the engine can swallow it."""


class CrashAfter:
    """Wrap a provider and crash the "process" on call number ``n + 1``. For resume demos/tests."""

    def __init__(self, inner: Any, n: int) -> None:
        self.inner, self.n, self.calls = inner, n, 0
        self.name = getattr(inner, "name", "wrapped")

    async def complete(self, request: CompletionRequest) -> Completion:
        if self.calls >= self.n:
            raise SimulatedCrash(f"simulated crash before call {self.calls + 1}")
        self.calls += 1
        return await self.inner.complete(request)


class MockProvider:
    name = "mock"

    def __init__(self, *, latency: float = 0.0, fault_rate: float = 0.0, subtle_fraction: float = 0.25,
                 seed: int = 0) -> None:
        if not 0.0 <= fault_rate <= 1.0:
            raise ValueError("fault_rate must be in [0, 1]")
        self.latency = latency
        self.fault_rate = fault_rate
        self.subtle_fraction = subtle_fraction
        self.seed = seed
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        if self.latency:
            # Simulated network/model latency: what makes parallel scheduling measurable offline.
            await asyncio.sleep(self.latency)
        if request.agent == "planner":
            text, calls = self._planner(request), []
        elif request.agent == "critic":
            text, calls = self._critic(request), []
        else:
            text, calls = self._worker(request)
        usage = Usage(estimate_tokens(request_text(request)),
                      estimate_tokens(text + "".join(json.dumps(c.arguments) for c in calls)))
        return Completion(text=text, tool_calls=calls, usage=usage,
                          stop_reason="tool_use" if calls else "end_turn", model="mock-1")

    # ================================================================ planner

    def _planner(self, request: CompletionRequest) -> str:
        prompt = request.messages[0].content
        goal = prompt.split("GOAL:\n", 1)[-1].split("\n\nWORKSPACE FILES:", 1)[0].strip()
        files: list[tuple[str, str | None]] = []
        for line in prompt.split("WORKSPACE FILES:\n", 1)[-1].split("\n\nROLES:", 1)[0].splitlines():
            m = _MANIFEST_RE.match(line.strip())
            if m:
                files.append((m.group(1), m.group(3)))
        low = goal.lower()
        if any(w in low for w in ("csv", "stats", "statistics")) and any(f.endswith(".csv") for f, _ in files):
            tasks = self._plan_csv(goal, files)
        elif "report" in low and any(f.endswith((".md", ".txt")) for f, _ in files):
            tasks = self._plan_report(goal, files)
        elif _expressions(goal):
            tasks = self._plan_calc(goal)
        else:
            tasks = [{"id": "respond", "description": goal, "role": "generalist", "tools": [],
                      "acceptance": ["Directly addresses the goal"]}]
        return "Here is the plan.\n```json\n" + json.dumps({"tasks": tasks}, indent=1) + "\n```"

    @staticmethod
    def _plan_report(goal: str, files: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
        texts = [f for f, _ in files if f.endswith((".md", ".txt")) and not f.startswith("output/")]
        # Prefer a folder the goal names ("from the local docs folder").
        named = [f for f in texts if f.split("/")[0].lower() in goal.lower() and "/" in f]
        docs = named or texts
        named_out = re.search(r"\b(output/[\w./-]+\.md)\b", goal)
        out_path = named_out.group(1) if named_out else "output/report.md"
        tasks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in docs:
            tid = "summarize_" + _slug(PurePosixPath(path).stem)
            while tid in seen:
                tid += "_x"
            seen.add(tid)
            tasks.append({
                "id": tid, "role": "researcher", "tools": ["read_file"], "deps": [],
                "description": f"Read {path} and write a faithful summary: Source, Title, Summary and Key terms lines.",
                "acceptance": ["The summary only states things the source says", "It cites the source file"],
                "checks": [{"type": "contains", "value": f"Source: {path}"}, {"type": "contains", "value": "Title: "},
                           {"type": "min_length", "value": 60}],
            })
        summaries = [t["id"] for t in tasks]
        tasks.append({
            "id": "outline", "role": "writer", "tools": [], "deps": summaries,
            "description": "Order the summaries into a report outline: one '## <title>' heading per source.",
            "acceptance": ["Every source has exactly one section"],
            "checks": [{"type": "regex", "pattern": r"^## \S"}],
        })
        tasks.append({
            "id": "write_report", "role": "writer", "tools": ["write_file"], "deps": ["outline", *summaries],
            "description": f"Write the final markdown report to {out_path}, following the outline, citing each source.",
            "acceptance": ["The report follows the outline and cites every source"],
            "checks": [{"type": "file_exists", "path": out_path}, {"type": "contains", "value": out_path}],
        })
        return tasks

    @staticmethod
    def _plan_csv(goal: str, files: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
        csvs = [(f, h) for f, h in files if f.endswith(".csv")]
        path, header = next(((f, h) for f, h in csvs if f in goal), csvs[0])
        columns = [c.strip() for c in (header or "").split(",") if c.strip()]
        return [
            {"id": "profile_data", "role": "analyst", "tools": ["python_eval"], "deps": [],
             "description": f"Load {path}; report columns, which columns are numeric, and the row count as JSON.",
             "checks": [{"type": "json_keys", "keys": ["columns", "numeric", "rows"]}]},
            {"id": "compute_stats", "role": "analyst", "tools": ["python_eval"], "deps": ["profile_data"],
             "description": f"Compute mean, median, stdev, min, max and count for every numeric column of {path}; "
                            "summarise categorical columns by distinct count. Output one JSON object keyed by column.",
             "acceptance": ["Numbers are computed with a tool, not estimated"],
             "checks": [{"type": "json_keys", "keys": columns}] if columns else []},
            {"id": "chart", "role": "analyst", "tools": ["python_eval", "write_file"], "deps": ["profile_data"],
             "description": f"Draw an SVG bar chart of the per-column totals of the numeric columns of {path} "
                            "and save it to output/chart.svg.",
             "checks": [{"type": "file_exists", "path": "output/chart.svg"}]},
            {"id": "write_summary", "role": "writer", "tools": ["write_file"], "deps": ["compute_stats", "chart"],
             "description": "Write output/stats_report.md: a markdown table of the statistics plus the chart.",
             "acceptance": ["Every numeric column appears in the table"],
             "checks": [{"type": "file_exists", "path": "output/stats_report.md"}]},
        ]

    @staticmethod
    def _plan_calc(goal: str) -> list[dict[str, Any]]:
        exprs = _expressions(goal)
        tasks: list[dict[str, Any]] = [
            {"id": f"calc_{i}", "role": "generalist", "tools": ["calculator"], "deps": [],
             "description": f"Evaluate exactly: {e}. Reply with the number only.",
             "checks": [{"type": "regex", "pattern": r"^-?\d+(\.\d+)?$"}]}
            for i, e in enumerate(exprs, 1)
        ]
        tasks.append({
            "id": "combine", "role": "generalist", "tools": [], "deps": [t["id"] for t in tasks],
            "description": "List every expression with its value, one per line, as `expression = value`. "
                           "Expressions: " + " | ".join(f"{t['id']}: {e}" for t, e in zip(tasks, exprs)),
            "checks": [{"type": "contains", "value": f"{e} = "} for e in exprs],
        })
        return tasks

    # ================================================================ worker

    def _fault(self, task_id: str, attempt: int, supports_subtle: bool) -> str | None:
        rng = random.Random(f"{self.seed}|{task_id}|{attempt}")
        if rng.random() >= self.fault_rate:
            return None
        if rng.random() < self.subtle_fraction:
            return "subtle" if supports_subtle else None
        return "obvious"

    def _worker(self, request: CompletionRequest) -> tuple[str, list[ToolCall]]:
        prompt = request.messages[0].content
        spec = json.loads(prompt.split("TASK_SPEC: ", 1)[1].split("\n", 1)[0])
        tid, attempt = spec["id"], int(spec["attempt"])
        task_line = re.search(r"^TASK: (.*)$", prompt, re.MULTILINE)
        description = task_line.group(1) if task_line else ""
        deps = dict(_DEP_RE.findall(prompt))
        results = [m for m in request.messages if m.role == "tool"]
        step = len([m for m in request.messages if m.role == "assistant"])

        def call(name: str, **args: Any) -> tuple[str, list[ToolCall]]:
            return "", [ToolCall(f"call_{tid}_{attempt}_{step}", name, args)]

        if tid.startswith("summarize_"):
            return self._w_summarize(tid, attempt, description, results, call)
        if tid == "outline":
            fault = self._fault(tid, attempt, False)
            sections = [_parse_summary(v) for k, v in sorted(deps.items())]
            if fault == "obvious":
                return "Sections: " + ", ".join(s["title"] for s in sections), []
            return "\n".join(f"## {s['title']}\n(source: {s['source']})" for s in sections), []
        if tid == "write_report":
            out_path = re.search(r"report to (\S+?), following", description)
            path = out_path.group(1) if out_path else "output/report.md"
            fault = self._fault(tid, attempt, False)
            if step == 0 and fault != "obvious":
                return call("write_file", path=path, content=_render_report(deps))
            if results and results[-1].is_error:
                return f"Could not write {path}: {results[-1].content}", []
            n = sum(1 for k in deps if k.startswith("summarize_"))
            return f"Wrote {path} covering {n} sources.", []
        if tid in ("profile_data", "compute_stats", "chart"):
            return self._w_csv(tid, attempt, description, results, step, call)
        if tid == "write_summary":
            fault = self._fault(tid, attempt, False)
            if step == 0 and fault != "obvious":
                return call("write_file", path="output/stats_report.md",
                            content=_render_stats(json.loads(deps.get("compute_stats", "{}"))))
            if results and results[-1].is_error:
                return f"Could not write output/stats_report.md: {results[-1].content}", []
            return "Wrote output/stats_report.md with the statistics table and the chart.", []
        if tid.startswith("calc_"):
            expr = description.split("Evaluate exactly: ", 1)[-1].rsplit(". Reply", 1)[0]
            if step == 0:
                return call("calculator", expression=expr)
            value = results[-1].content
            if results[-1].is_error:
                return f"Could not evaluate: {value}", []
            fault = self._fault(tid, attempt, True)
            if fault == "obvious":
                return f"The answer is approximately {value}.", []
            if fault == "subtle":
                return _bump(value), []
            return value, []
        if tid == "combine":
            listed = description.split("Expressions: ", 1)[-1].split(" | ")
            pairs = [item.split(": ", 1) for item in listed if ": " in item]
            return "\n".join(f"{expr} = {deps.get(dep, '?')}" for dep, expr in pairs), []
        return ("MockProvider has no scripted behaviour for this goal; run with a real provider "
                "(--provider anthropic or --provider openai) for open-ended goals."), []

    def _w_summarize(self, tid: str, attempt: int, description: str, results: list[Message],
                     call: Any) -> tuple[str, list[ToolCall]]:
        path = description.split("Read ", 1)[1].split(" and write", 1)[0]
        if not results:
            return call("read_file", path=path)
        if results[-1].is_error:
            return f"Could not read {path}: {results[-1].content}", []
        summary = _summarize(path, results[-1].content)
        fault = self._fault(tid, attempt, True)
        if fault == "obvious":
            # Forgets to cite and to state the title: the contains-checks catch this.
            return summary.split("Summary: ", 1)[-1], []
        if fault == "subtle":
            # Plausible but wrong title: passes every check, fails the ground truth.
            return re.sub(r"^Title: .*$", "Title: Project notes", summary, count=1, flags=re.MULTILINE), []
        return summary, []

    def _w_csv(self, tid: str, attempt: int, description: str, results: list[Message], step: int,
               call: Any) -> tuple[str, list[ToolCall]]:
        path = re.search(r"(\S+\.csv)", description).group(1)  # type: ignore[union-attr]
        load = f"rows = list(csv.DictReader(read_text({path!r}).splitlines()))\n"
        if results and results[-1].is_error:
            return f"Could not complete {tid}: {results[-1].content}", []
        if tid == "profile_data":
            if step == 0:
                return call("python_eval", code=load + _PROFILE_CODE)
            return json.dumps(json.loads(results[-1].content)["result"]), []
        if tid == "compute_stats":
            if step == 0:
                return call("python_eval", code=load + _STATS_CODE)
            stats = json.loads(results[-1].content)["result"]
            fault = self._fault(tid, attempt, True)
            if fault == "obvious" and stats:
                stats.pop(list(stats)[-1])  # silently drops a column: json_keys check catches it
            elif fault == "subtle":
                for col in stats.values():
                    if "mean" in col:
                        col["mean"] = round(col["mean"] + 1.0, 6)  # wrong but well-formed
                        break
            return json.dumps(stats), []
        # chart
        if step == 0:
            return call("python_eval", code=load + _CHART_CODE)
        if step == 1:
            fault = self._fault(tid, attempt, False)
            if fault == "obvious":
                return "Chart drawn.", []  # claims success without saving the file
            svg = json.loads(results[-1].content)["result"]
            return call("write_file", path="output/chart.svg", content=svg)
        return "Saved a bar chart of numeric column totals to output/chart.svg.", []

    # ================================================================ critic

    @staticmethod
    def _critic(request: CompletionRequest) -> str:
        prompt = request.messages[0].content
        output = prompt.split("<<<output>>>\n", 1)[-1].rsplit("\n<<<end:output>>>", 1)[0]
        low = output.lower()
        if not output.strip():
            return json.dumps({"pass": False, "feedback": "The output is empty."})
        if low.startswith("could not") or "error:" in low:
            return json.dumps({"pass": False, "feedback": "The output reports a failure; complete the task."})
        if "approximately" in low or "i think" in low:
            return json.dumps({"pass": False, "feedback": "Give the exact computed value, not an estimate."})
        return json.dumps({"pass": True, "feedback": ""})


# -------------------------------------------------------------------- helpers

_PROFILE_CODE = """
cols = list(rows[0].keys()) if rows else []
def isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
numeric = [c for c in cols if rows and all(isnum(r[c]) for r in rows)]
result = {"columns": cols, "numeric": numeric, "rows": len(rows)}
"""

_STATS_CODE = """
result = {}
for c in (rows[0].keys() if rows else []):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[c]))
        except ValueError:
            vals = None
            break
    if vals:
        result[c] = {"mean": round(statistics.fmean(vals), 6), "median": round(statistics.median(vals), 6),
                     "stdev": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                     "min": min(vals), "max": max(vals), "count": len(vals)}
    else:
        result[c] = {"type": "categorical", "distinct": len(set(r[c] for r in rows))}
"""

_CHART_CODE = """
totals = {}
for c in (rows[0].keys() if rows else []):
    try:
        totals[c] = sum(float(r[c]) for r in rows)
    except ValueError:
        pass
top = max(totals.values()) if totals else 1.0
top = top or 1.0
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="260">' % (80 * len(totals) + 40)]
for i, (c, v) in enumerate(totals.items()):
    h = 200.0 * v / top
    x = 30 + 80 * i
    parts.append('<rect x="%d" y="%.1f" width="50" height="%.1f" fill="#4a7bd0"/>' % (x, 220 - h, h))
    parts.append('<text x="%d" y="240" font-size="11">%s</text>' % (x, c))
    parts.append('<text x="%d" y="%.1f" font-size="10">%.0f</text>' % (x, 215 - h, v))
parts.append('</svg>')
result = "\\n".join(parts)
"""


def _summarize(path: str, text: str) -> str:
    lines = text.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), PurePosixPath(path).stem)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() and not p.lstrip().startswith("#")]
    firsts = []
    for p in paragraphs[:3]:
        sentence = re.split(r"(?<=[.!?])\s", " ".join(p.split()), maxsplit=1)[0]
        firsts.append(sentence)
    words = [w for w in re.findall(r"[a-z][a-z-]{3,}", text.lower()) if w not in _STOPWORDS]
    terms = [w for w, _ in Counter(words).most_common(5)]
    return f"Source: {path}\nTitle: {title}\nSummary: {' '.join(firsts)}\nKey terms: {', '.join(terms)}"


def _parse_summary(text: str) -> dict[str, str]:
    fields = {"source": "unknown", "title": "Untitled", "summary": text.strip(), "terms": ""}
    for line in text.splitlines():
        for key, label in (("source", "Source: "), ("title", "Title: "), ("summary", "Summary: "),
                           ("terms", "Key terms: ")):
            if line.startswith(label):
                fields[key] = line[len(label):].strip()
    return fields


def _render_report(deps: dict[str, str]) -> str:
    summaries = [_parse_summary(v) for k, v in sorted(deps.items()) if k.startswith("summarize_")]
    all_terms = Counter(t.strip() for s in summaries for t in s["terms"].split(",") if t.strip())
    out = ["# Report", "", f"Synthesised from {len(summaries)} source documents.", ""]
    if all_terms:
        out += ["Recurring themes: " + ", ".join(t for t, _ in all_terms.most_common(6)) + ".", ""]
    for s in summaries:
        out += [f"## {s['title']}", "", s["summary"], "", f"_Source: {s['source']}_", ""]
    return "\n".join(out)


def _render_stats(stats: dict[str, Any]) -> str:
    out = ["# Statistics", "", "| column | mean | median | stdev | min | max | count |",
           "|---|---|---|---|---|---|---|"]
    for col, s in stats.items():
        if "mean" in s:
            out.append(f"| {col} | {s['mean']:.4f} | {s['median']:.4f} | {s['stdev']:.4f} | "
                       f"{s['min']:g} | {s['max']:g} | {s['count']} |")
    cats = [f"{c} ({s['distinct']} distinct)" for c, s in stats.items() if s.get("type") == "categorical"]
    if cats:
        out += ["", "Categorical columns: " + ", ".join(cats) + "."]
    out += ["", "![column totals](chart.svg)", ""]
    return "\n".join(out)


_EXPR_RE = re.compile(r"^[\d\s.+\-*/()%a-z_,]*\d[\d\s.+\-*/()%a-z_,]*$")


def _expressions(goal: str) -> list[str]:
    """Expressions in goals like 'Calculate: 2+3; sqrt(16)*4'."""
    m = re.search(r"(?:calculate|compute|evaluate)[^:]*:\s*(.+)$", goal, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    parts = [p.strip().rstrip(".") for p in m.group(1).split(";")]
    return [p for p in parts if p and _EXPR_RE.match(p) and len(p) < 200]


def _bump(value: str) -> str:
    try:
        n = float(value)
    except ValueError:
        return value
    n += 1
    return str(int(n)) if n.is_integer() else str(n)
