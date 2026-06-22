# Examples

A fully offline, deterministic demo of `agentbisect` — no network, no API keys.

## `refund_regression.py`

Simulates a support agent whose answer must mention the refund policy. The agent is a
deterministic [`FakeAgent`](../src/agentbisect/agent/fake.py) and the judge is an
[`AssertionOracle`](../src/agentbisect/oracle.py), so the whole run is reproducible.

The scenario: a prompt history of five revisions where an edit in the middle drops the
"refund policy" clause — a regression. The demo captures the original (good) run, then
bisects the model/prompt list to pinpoint the **first bad change**.

Run it:

```bash
python examples/refund_regression.py
```

Expected output (deterministic):

```
captured baseline: refund=yes
bisecting 5 candidates over axis 'prompt-sim'...
first bad change: rev-2  (last good: rev-1)
behavioral diff: final output 'refund=yes' -> 'refund=no'
minimal repro: 1 step(s)
probes used: 4
```

The same flow is available through the CLI; see the project config
[`project.py`](./project.py) and:

```bash
agentbisect capture --config examples/project.py --out /tmp/bundle --label demo
```
