"""Offline, deterministic agentbisect demo: find the prompt edit that dropped a clause.

Run with::

    python examples/refund_regression.py

No network and no API keys: the agent is a deterministic :class:`FakeAgent` and the
judge is an :class:`AssertionOracle`. The demo captures a good baseline run, then
bisects an ordered list of prompt revisions to pinpoint the first revision that removed
the "refund policy" clause (a regression).
"""

from __future__ import annotations

from agentbisect.agent.fake import FakeAgent
from agentbisect.capture import capture
from agentbisect.driver import run_bisection
from agentbisect.oracle import AssertionOracle
from agentbisect.types import AgentConfig, Candidate, RunBundle, Trace


def _config(prompt: str) -> AgentConfig:
    """Build a config whose FakeAgent answer reports whether 'refund' is in the prompt.

    The ``{prompt_has:refund}`` marker resolves to ``yes``/``no`` based on the system
    prompt, so a prompt edit that drops the clause deterministically changes the answer.
    """
    return AgentConfig(
        system_prompt=prompt,
        model="support-bot",
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund={prompt_has:refund}",
        },
    )


def main() -> None:
    # An ordered prompt history (old -> new). rev-2 is where the refund clause is dropped.
    prompt_revisions = [
        "You are support. Always state the refund policy clearly.",  # rev-0 good
        "You are support. Always state the refund policy. Be friendly.",  # rev-1 good
        "You are support. Be friendly and concise.",  # rev-2 BAD (clause removed)
        "You are support. Be concise.",  # rev-3 bad
        "You are support.",  # rev-4 bad
    ]

    agent = FakeAgent()

    # Capture the baseline (oldest, good) run for deterministic tool replay.
    bundle: RunBundle = capture(
        agent,
        _config(prompt_revisions[0]),
        lambda tool, args: f"{tool}-doc",
        label="demo-baseline",
    )
    print(f"captured baseline: {bundle.trace.final_output}")

    # Build the candidate list: one per revision, overriding only the prompt.
    candidates = [
        Candidate(
            axis="prompt-sim",
            ref=f"rev-{i}",
            order=i,
            config=_config(prompt),
        )
        for i, prompt in enumerate(prompt_revisions)
    ]

    # Oracle: GOOD iff the answer still reports the refund clause present.
    oracle = AssertionOracle(lambda trace, _b: trace.final_output == "refund=yes")

    print(f"bisecting {len(candidates)} candidates over axis 'prompt-sim'...")
    outcome = run_bisection(agent, bundle, candidates, oracle)
    result = outcome.result

    if result.first_bad is not None and result.last_good is not None:
        print(f"first bad change: {result.first_bad.ref}  (last good: {result.last_good.ref})")
        if outcome.behavioral_diff is not None and outcome.behavioral_diff.final_output_changed:
            bd = outcome.behavioral_diff
            print(f"behavioral diff: final output {bd.left_final!r} -> {bd.right_final!r}")
        repro: Trace | None = outcome.minimal_repro
        if repro is not None:
            print(f"minimal repro: {len(repro.steps)} step(s)")
    else:
        print("result: ambiguous range (no single first-bad change isolated)")
    print(f"probes used: {result.probes}")


if __name__ == "__main__":
    main()
