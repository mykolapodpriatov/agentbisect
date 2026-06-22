# agentbisect

> git-bisect for agent regressions — replay a captured failing run across prompt/model/tool versions to find the exact change that broke behavior.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Treats an agent run as a function of (system prompt, model id/params, tool schemas, retrieval snapshot). It captures a run as a versioned bundle with recorded tool I/O for deterministic replay, then binary-searches an ordered set of snapshots to isolate the first change that broke behavior.

## Why

When an agent regresses, the cause is buried across prompts, model versions, and tool changes. This finds the culprit automatically instead of by hand.

## Features

- Capture a run as a versioned bundle: prompt, model id/params, tool schemas, recorded tool I/O
- Deterministic replay engine that mocks tool calls from the recorded trace
- Automated bisection over an ordered set of versioned snapshots
- LLM-judge or assertion oracle to drive good/bad/skip automatically
- Outputs first-bad-change + responsible axis + minimal reproducing trace + behavioral diff

## How it works

Record a failing run once. Point agentbisect at an ordered history (git commits of prompt/tool files, or a model-version list); it replays each candidate with tool calls mocked so only the variable under test changes, judged by an oracle, until it isolates the breaking change.

## Tech stack

- Python
- Anthropic / OpenAI SDKs
- Ollama (judge)
- Pydantic
- click

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Run-capture bundle format + deterministic mocked replay
- [ ] Bisection driver with LLM-judge/assertion oracle
- [ ] First-bad-change report with minimal repro + behavioral diff
- [ ] Use promptfoo as an oracle; parallel bisection workers

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
