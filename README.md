# taskforge

**A minimal, framework-free multi-agent orchestration engine: a planner turns a goal into a validated task DAG, workers run it in parallel with sandboxed tools, and a critic sends bad output back for revision. Budgets, checkpoints and traces are built in.**

Pure standard library, under 3,000 lines (half of them comments) you can read in an afternoon. It runs fully offline with a deterministic mock model, and plugs into Anthropic, OpenAI or a local Ollama when you want a real one.

```
pip install -e .        # Python 3.11+, zero runtime dependencies
taskforge run "Research and write a report from the local docs folder" -w examples/workspaces/docs_report
```

## Why

Multi-agent frameworks hide the parts that decide whether an agent system works in production:
*what happens when the plan has a cycle, when a tool call has bad arguments, when the run crashes at
task 7 of 12, when the bill gets too big, when a model says "done" but never wrote the file.* taskforge is
those parts and nothing else, written out plainly:

- **Planning is untrusted input.** The planner's DAG is validated (no cycles, every dependency exists,
  known roles, valid checks). If it fails, the exact error (`dependency cycle: a -> b -> a`) goes back to the
  planner for a re-plan.
- **Parallelism comes from the graph.** Independent tasks run concurrently on asyncio with a hard
  concurrency cap. Data moves only along DAG edges, through a shared blackboard.
- **Verification is a separate agent.** A critic runs cheap deterministic checks first (file exists, JSON
  keys, regex), then an LLM judgement. Rejected output goes back to the worker with specific feedback, up to a
  set number of revisions.
- **Failures are contained.** A failed task skips only its own downstream tasks. A budget overrun is a hard
  stop. A crash resumes from the last completed task.
- **Everything is observable.** Each LLM call, tool call, approval, verdict and checkpoint is written as a
  JSONL event, and `taskforge timeline` draws the run as a Gantt chart in the terminal.

## Quickstart

```bash
git clone https://github.com/lijo123ferdinand/taskforge && cd taskforge
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'          # add ',anthropic' for the Anthropic provider
pytest -q                        # 87 passed
```

Copy an example workspace so the run's outputs don't land in the repo, then plan and run. The output below is
pasted from a real run. `--fault-rate 0.3 --seed 8` makes the mock model make a mistake so the critic has
something to catch:

```
$ cp -r examples/workspaces/csv_stats /tmp/csv_stats && cd /tmp/csv_stats
$ taskforge plan "Compute and chart stats from the CSV in data/"
L0 profile_data             role=analyst    deps=[] tools=['python_eval']
L1 compute_stats            role=analyst    deps=['profile_data'] tools=['python_eval']
L1 chart                    role=analyst    deps=['profile_data'] tools=['python_eval', 'write_file']
L2 write_summary            role=writer     deps=['compute_stats', 'chart'] tools=['write_file']

$ taskforge run "Compute and chart stats from the CSV in data/" --latency 0.05 --fault-rate 0.3 --seed 8
task           0.00s                                       1.00s
(planner)      |####............................................|  done     1 att  0 tools    775 tok
profile_data   |...###########..................................|  done     1 att  1 tools    845 tok
compute_stats  |..............##########################........|  done     2 att  2 tools   3018 tok  critic rejected x1
chart          |..............###############...................|  done     1 att  1 tools   1418 tok
write_summary  |........................................########|  done     1 att  1 tools   1083 tok
run completed: 4 done, 0 failed, 0 skipped in 1.00s; 13 LLM calls, 7139 tokens
```

`compute_stats` and `chart` ran side by side. The first `compute_stats` attempt silently dropped a column; the
critic caught it from the trace (`.taskforge/trace.jsonl`):

```json
{"type": "critic_verdict", "task": "compute_stats", "attempt": 0, "passed": false, "source": "checks",
 "feedback": "JSON output is missing keys ['discount_pct']"}
```

The run produced `output/stats_report.md` (values computed in the python sandbox) and `output/chart.svg`:

```
| column | mean | median | stdev | min | max | count |
|---|---|---|---|---|---|---|
| units | 63.3500 | 62.5000 | 30.0278 | 12 | 114 | 40 |
| unit_price | 15.6152 | 14.5000 | 5.3886 | 9.99 | 24.75 | 40 |
| discount_pct | 13.4750 | 16.0000 | 7.1539 | 1 | 24 | 40 |
```

The docs goal fans out to one summariser per document:

```
$ taskforge run "Research and write a report from the local docs folder" -w examples/workspaces/docs_report --latency 0.05
task                     0.00s                                       0.70s
(planner)                |####............................................|  done     1 att  0 tools   1499 tok
summarize_caching        |...##############...............................|  done     1 att  1 tools    930 tok
summarize_consistency    |...##############...............................|  done     1 att  1 tools    967 tok
summarize_observability  |...##############...............................|  done     1 att  1 tools    985 tok
summarize_queues         |...##############...............................|  done     1 att  1 tools    924 tok
summarize_rate_limiting  |.................############...................|  done     1 att  1 tools    976 tok
summarize_retries        |.................############...................|  done     1 att  1 tools    935 tok
outline                  |.............................########...........|  done     1 att  0 tools   1049 tok
write_report             |.....................................###########|  done     1 att  1 tools   2997 tok
run completed: 8 done, 0 failed, 0 skipped in 0.70s; 24 LLM calls, 11262 tokens
```

The budget, crash/resume and approval features look like this. All three are real runs:

```
$ taskforge run "Research and write a report from the local docs folder" --max-tokens 5000 -q
{ "status": "budget_exhausted", ... }            # hard stop; in-flight tasks put back to pending
$ taskforge resume --max-tokens 100000 -q        # finishes only what was left; spend keeps counting

$ python examples/crash_and_resume.py
crashed: simulated crash before call 10
checkpoint has 2/8 tasks done: ['summarize_caching', 'summarize_consistency']
resumed run: status=completed, model calls after resume=17

$ taskforge run "..." --approve write_file          # asks y/N on the terminal before every write
```

Use `taskforge timeline .taskforge/trace.jsonl` to redraw any past run. `examples/quickstart.py` shows the
library API: an `Orchestrator`, a `PolicyApprover` and a `Budget`.

### Real models

```bash
export ANTHROPIC_API_KEY=...            # never committed; read from the environment
taskforge run "..." --provider anthropic                       # default model claude-sonnet-5-5
taskforge run "..." --provider openai --model llama3.1         # Ollama at http://localhost:11434/v1
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... taskforge run "..." --provider openai --model <id>
taskforge run "..." --provider anthropic --max-cost 0.50 --price-in <usd/Mtok> --price-out <usd/Mtok>
```

## Architecture

```
                 goal + workspace manifest (files, CSV headers)
                                   |
                                   v
     +------------------------- Planner --------------------------+
     |  LLM -> JSON plan -> TaskGraph.from_dict (validate)         |
     |  invalid? feed exact error back, re-plan (max 3)            |
     +------------------------------+------------------------------+
                                    | validated DAG
                                    v
  +----------------------------- Scheduler (one coroutine owns all state) ------------------------+
  |  ready = deps all done  ->  launch while running < max_concurrency  ->  asyncio.wait(FIRST)  |
  |                                                                                                |
  |   +---------- task ----------+     +---------- task ----------+     +---------- task ------+   |
  |   | Worker (role prompt,     |     | Worker                   |     | Worker               |   |
  |   |   tool allow-list)       |     |  tool loop:              |     |                      |   |
  |   |   <-> ToolRegistry       |     |   allow-list? approval?  |     |                      |   |
  |   |       schema-validate    |     |   schema? sandbox?       |     |                      |   |
  |   | Critic: checks -> LLM    |     | Critic                   |     | Critic               |   |
  |   |   fail -> revise (<=N)   |     |                          |     |                      |   |
  |   +------------+-------------+     +------------+-------------+     +----------+-----------+   |
  |                | result                         |                              |               |
  |                v                                v                              v               |
  |        Blackboard  result:<task>  --->  downstream prompts (only along DAG edges)              |
  |        checkpoint.json (atomic, after every finished task)                                     |
  +------------------------------------------------------------------------------------------------+
        every LLM call -> LLMGateway: budget pre-check -> provider (retry 429/5xx) -> charge -> trace
        every event    -> trace.jsonl  ->  taskforge timeline
```

| Module | Responsibility |
|---|---|
| `dag.py` | `Task`, `TaskGraph`: validation (iterative DFS cycle detection with the cycle path), ready set, levels, descendants |
| `agents.py` | `LLMGateway` (budget, retries, tracing), `Planner`, `Worker` (tool loop), `Critic` |
| `engine.py` | `Orchestrator`: scheduling, revision loop, failure propagation, budget stop, checkpoint/resume |
| `tools/` | registry + JSON-schema validation, sandboxed `read_file`/`write_file`/`list_dir`, AST `calculator`, `python_eval` subprocess sandbox, opt-in `web_fetch` |
| `providers/` | `Provider` protocol; `MockProvider`; `AnthropicProvider`; `OpenAICompatProvider` (urllib) |
| `budget.py`, `approval.py`, `blackboard.py`, `checkpoint.py`, `trace.py`, `timeline.py` | the rest |

## Design decisions and trade-offs

- **No framework, no dependencies.** The schema validator, HTTP client and sandbox are written on the stdlib.
  That costs some lines, but the engine is easy to audit and easy to vendor.
- **One coroutine owns state; workers only return values.** Checkpoints are written between `asyncio.wait`
  wake-ups, so they can never capture a half-applied result. No locks are needed around task state.
- **Bounded launch instead of a semaphore.** A semaphore would mark tasks as "started" while they wait in a
  queue, and the timeline would show work that isn't happening. taskforge only launches what can run.
- **Checks before LLM judgement.** A mechanical check is free and exact. When one fails, the critic skips the
  LLM call and sends the precise failure as feedback. An LLM verdict it can't parse *fails open*, because the
  deterministic gate already passed and failing closed would spend the revision budget on a formatting bug.
- **Allow-lists narrow and never widen.** The effective tools for a task are `role.tools ∩ task.tools`, so a
  prompt-injected planner can't give the writer a Python sandbox. A disallowed call returns an error to the
  model and is traced as `tool_denied`.
- **Tool errors are results, not exceptions.** Bad arguments, sandbox escapes and timeouts all come back to
  the model as `ERROR: ...` with the JSON path that was wrong, so it can correct itself.
- **Hard budget stop with a bounded overshoot.** Each call is pre-checked against an estimate and charged on
  the reported usage. The worst overshoot is one call per concurrent worker. Reserving `max_tokens` up front
  would remove the overshoot but starve parallel workers. No price table ships, because stale prices are worse
  than none: you pass `--price-in/--price-out`.
- **`web_fetch` is unregistered by default.** A model can't be prompt-injected into calling a tool it was
  never shown. Even when enabled, it allows only http(s) and supports a host allow-list.
- **The mock model is a scripted policy, not a fake engine.** It reads the same prompts a real model gets
  and returns real plans and tool calls. Every plan is validated and every call is executed, sandboxed and
  checked, so tests exercise the real engine paths.

## Benchmarks

Reproduce with `python benchmarks/bench.py --json benchmarks/results.json` (about 40 s; raw numbers are in
`benchmarks/results.json`). The figures below were measured on an Apple Silicon Mac with Python 3.14. **Read them
for what they are:** they come from the offline MockProvider. The scheduling numbers use a *simulated* 50 ms per
LLM call, so they measure the scheduler, not any model. The critic numbers use *seeded fault injection* and are
graded by an independent ground-truth grader that recomputes the answers from the input files.

**Parallel vs sequential** (wall time including planning, median of 3):

| DAG | tasks | depth | LLM calls | conc=1 | conc=2 | conc=4 | conc=8 | speedup @4 |
|---|---|---|---|---|---|---|---|---|
| docs_report | 8 | 3 | 24 | 1.287s | 0.8s | 0.637s | 0.482s | 2.02x |
| csv_stats | 4 | 3 | 12 | 0.917s | 0.725s | 0.749s | 0.712s | 1.23x |
| calc | 6 | 2 | 12 | 0.649s | 0.444s | 0.327s | 0.218s | 1.99x |

Speedup is bounded by the DAG's shape. `csv_stats` has only one parallel level, two tasks wide, so concurrency
beyond 2 can't help. Its runs also include real `python_eval` subprocess time, which is why conc=4 is no faster
than conc=2.

**With vs without the critic** (each worker attempt faults with p = 0.2; 25% of faults are *subtle*, meaning
well-formed but wrong, so no check can catch them; up to 2 revisions; 20 seeds x 5 goals = 100 runs per mode):

| mode | end-to-end success | avg LLM calls | avg tokens | docs | csv | calc |
|---|---|---|---|---|---|---|
| no critic | 39.0% | 11.1 | 4637 | 20% | 50% | 42% |
| critic | **82.0%** | 15.0 | 5836 | 75% | 90% | 82% |

The critic more than doubles end-to-end success, from 39% to 82%, for about 35% more LLM calls and 26% more
tokens. It doesn't reach 100%: the subtle faults (a plausible but wrong title, a mean off by one) pass every
check, and only a grader with ground truth catches them. The docs DAG has 8 tasks that can each fault, so it
benefits most from review.

## Limitations

- **The python sandbox is not a security boundary.** It uses process isolation (`python -I`), an AST screen
  (no imports, dunders, frame attributes or `str.format`), restricted builtins, CPU rlimits and a wall-clock
  kill. That stops a sloppy model and makes deliberate escapes harder. For untrusted workloads, run taskforge
  inside a container or VM. `RLIMIT_AS` is not enforced on macOS.
- **The benchmarks use a mock model.** They show the engine's scheduling and revision mechanics, not how well
  Claude or Llama plans. A real-model eval suite is on the roadmap.
- **The MockProvider only knows three goal families** (report from docs, CSV stats and chart, arithmetic).
  Any other goal gets a one-task plan that says to use a real provider.
- **Resume granularity is one task.** In-flight tasks are redone from scratch. Tokens spent by calls that were
  in flight at crash time are not in the checkpoint, so resumed spend can under-count by those calls.
- **Plans are static after validation.** There is no dynamic re-planning when a branch fails; its dependents
  are skipped. `resume --retry-failed` re-runs failed and skipped tasks.
- **Single process.** Workers are asyncio tasks, not distributed. Concurrency helps I/O-bound LLM calls, not
  CPU-bound tools.

## Roadmap

- Dynamic re-planning: let the planner patch the DAG when a task fails permanently.
- Real-model eval suite (Claude, Llama via Ollama) on the same graded tasks, with cost per success.
- Streaming token usage and a pre-call reservation mode for zero-overshoot budgets.
- Per-tool rate limits and caching of idempotent tool results across revisions.
- Container-backed `python_eval` (Docker / nsjail) as a drop-in tool.
- An MCP client so tools can come from any MCP server.

## Development

```bash
pytest -q                                   # 87 tests, ~8 s
python benchmarks/bench.py                  # regenerate benchmark tables
python examples/quickstart.py
python examples/crash_and_resume.py
```

MIT licensed. Copyright 2026 Lijo Ferdinand.
