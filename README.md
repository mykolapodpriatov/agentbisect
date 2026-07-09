# agentbisect

> git-bisect for agent regressions — replay a captured failing run across prompt/model/tool versions to find the exact change that broke behavior.

![status](https://img.shields.io/badge/status-alpha-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

`agentbisect` treats an agent run as a function of (system prompt, model id/params, tool
schemas, retrieval snapshot). It captures a run as a versioned bundle with recorded tool
I/O for **deterministic replay**, then binary-searches an ordered set of snapshots —
driven by an LLM-judge or assertion oracle — to isolate the first change that broke
behavior, and emits a minimal reproducing trace plus a side-by-side behavioral diff.

## Why

When an agent regresses, the cause is buried across prompt commits, model bumps, and
tool-schema changes. Running a per-version eval tells you *which* versions fail, but not
*which change* is responsible. `agentbisect` does the bisection for you and points at the
single culprit.

## How it works

1. **Capture** a failing run into a portable `RunBundle`: the `AgentConfig` (prompt,
   model id + params, tool schemas, retrieval ref) and a recorded `Trace` whose tool
   outputs are keyed by `(tool, canonical args, occurrence)` — so repeated/retried calls
   replay unambiguously.
2. **Replay** a candidate config deterministically: tool calls are served from the
   recording (tools never re-execute) and every LLM step is forced to `temperature=0`,
   so a behavior change is attributable to the axis under test, not nondeterminism.
3. **Bisect** an ordered list of candidates (prompt git history, a model list, tool-schema
   versions) with git-bisect-compatible good/bad/skip semantics, until the first bad
   change is isolated.
4. **Explain**: report the first-bad change, the responsible axis, a minimal reproducing
   trace, and a behavioral diff between the last-good and first-bad runs.

### Correctness guarantees

- **Pure, exhaustively-tested bisect core.** Endpoints are validated first
  (`UntestableEndpointError` vs `NonMonotonicError` are distinct); a single first-bad is
  emitted **only** for a directly-adjacent good→bad transition, otherwise an *ambiguous
  range* is returned rather than a guessed culprit; flaky candidates are detected and
  reported, never silently resolved.
- **No verdict on fabricated tool output.** If a candidate diverges (calls a tool with no
  recorded match) or a `nearest` substitution is used, that result is **quarantined** as
  `skip`. `passthrough` (live re-execution) is an explicit opt-in and is surfaced in the
  report — never silent.
- **Deterministic, CI-safe judging.** The `LLMJudge` cache key folds in the judge's
  *resolved* model id (the concrete model string, not a `*-latest` alias), so the same
  candidate is never re-judged differently across machines.

## Install

Requires Python 3.11+.

```bash
pip install -e .                      # core
pip install -e ".[openai,anthropic]"  # optional judge backends
pip install -e ".[dev]"               # tests, ruff, mypy
```

## Quick start (offline, no API keys)

A complete deterministic demo lives in [`examples/`](./examples/):

```bash
python examples/refund_regression.py
```

```
captured baseline: refund=yes
bisecting 5 candidates over axis 'prompt-sim'...
first bad change: rev-2  (last good: rev-1)
behavioral diff: final output 'refund=yes' -> 'refund=no'
minimal repro: 1 step(s)
probes used: 4
```

### CLI

```bash
# Capture a run (project.py defines runner/config/tool_executor/oracle hooks).
agentbisect capture --config examples/project.py --out bundle/ --label case-1

# Bisect over prompt git history (old -> new) and report the first bad commit.
agentbisect bisect --bundle bundle/ --config examples/project.py \
    --axis prompt --over /path/to/repo:prompts/system.txt

# Bisect over a model list.
agentbisect bisect --bundle bundle/ --config examples/project.py \
    --axis model --over gpt-4o-mini,gpt-4o,gpt-4.1

# Replay a single override, or diff two captured bundles.
agentbisect replay --bundle bundle/ --config examples/project.py --override model=gpt-4o
agentbisect diff bundleA/ bundleB/
```

### Library

```python
from agentbisect import capture, run_bisection, AssertionOracle, ModelListAxis

bundle = capture(runner, config, tool_executor, label="case-1")
candidates = ModelListAxis(["model-a", "model-b", "model-c"]).candidates(bundle.config)
oracle = AssertionOracle(lambda trace, b: "refund policy" in trace.final_output)

outcome = run_bisection(runner, bundle, candidates, oracle)
print(outcome.result.first_bad)        # the culprit (or None if ambiguous)
print(outcome.behavioral_diff)         # last-good vs first-bad
print(outcome.minimal_repro)           # minimal reproducing trace
```

## Axes

A bisect runs over exactly one axis at a time; each candidate overrides a single field of
the captured config (single-axis isolation):

- `ModelListAxis(models)` — order = list order.
- `PromptGitAxis(repo_path, file_path, rev)` — order = git history (old→new) of a prompt file.
- `ToolSchemaAxis(versions)` — ordered tool-schema sets.
- `RetrievalAxis(snapshots)` — ordered retrieval snapshot refs.

## Judges

- `AssertionOracle` — a user predicate over the trace / final output.
- `LLMJudge` — a pinned judge prompt + model with a committable, deterministic verdict
  cache. Backends: OpenAI, Anthropic, Ollama, and an offline `FakeLLM` for tests.
- `FakeOracle` — scripted verdicts for tests/demos.

## Tech stack

- Python 3.11+ (fully typed; `mypy --strict`)
- Pydantic (immutable, validated models)
- Typer + Rich (CLI and reports)
- GitPython (prompt-history axis)
- OpenAI / Anthropic / Ollama (optional judge backends)

## Development

```bash
ruff check
ruff format --check
mypy src
pytest -q --cov=agentbisect
```

CI runs lint + format + type-check + tests on Python 3.11, 3.12, and 3.13.

## Roadmap

- [x] Run-capture bundle format + deterministic mocked replay
- [x] Pure bisection core with good/bad/skip + quarantine of fabricated tool output
- [x] First-bad-change report with minimal repro + behavioral diff
- [x] LLM-judge / assertion oracle; prompt-git and model-list axes
- [ ] Parallel candidate evaluation; HTML culprit report
- [ ] Adapter to drive a real agent framework end-to-end

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
