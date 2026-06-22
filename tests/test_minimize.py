"""Tests for trace minimization."""

from __future__ import annotations

from agentbisect.minimize import minimize
from agentbisect.types import LlmStep, ToolStep, Trace, Verdict


def _trace(n_steps: int, final: str) -> Trace:
    steps = tuple(LlmStep(index=i, role="assistant", content=f"step-{i}") for i in range(n_steps))
    return Trace(steps=steps, final_output=final)


def test_minimize_drops_redundant_trailing_steps() -> None:
    # The badness depends only on the first 2 steps existing; trailing ones are redundant.
    bad = _trace(6, "BAD")

    def verdict(t: Trace) -> Verdict:
        return Verdict.BAD if len(t.steps) >= 2 else Verdict.GOOD

    minimal = minimize(bad, verdict)
    assert len(minimal.steps) == 2
    assert verdict(minimal) is Verdict.BAD
    # Indices are re-numbered contiguously.
    assert [s.index for s in minimal.steps] == [0, 1]


def test_minimize_keeps_full_trace_when_all_needed() -> None:
    bad = _trace(4, "BAD")

    def verdict(t: Trace) -> Verdict:
        return Verdict.BAD if len(t.steps) == 4 else Verdict.GOOD

    minimal = minimize(bad, verdict)
    assert len(minimal.steps) == 4


def test_minimize_non_bad_input_returned_unchanged() -> None:
    good = _trace(3, "GOOD")

    def verdict(t: Trace) -> Verdict:
        return Verdict.GOOD

    minimal = minimize(good, verdict)
    assert len(minimal.steps) == 3


def test_minimize_is_bounded() -> None:
    bad = _trace(10, "BAD")
    calls = {"n": 0}

    def verdict(t: Trace) -> Verdict:
        calls["n"] += 1
        return Verdict.BAD  # always bad -> shrinks to the single-step floor

    minimal = minimize(bad, verdict, max_passes=3)
    # With a budget of 3 we can drop at most 3 trailing steps.
    assert len(minimal.steps) == 7
    # Call count is bounded: 1 (initial check) + up to max_passes.
    assert calls["n"] <= 1 + 3


def test_minimize_floor_is_single_step() -> None:
    bad = _trace(5, "BAD")

    def verdict(t: Trace) -> Verdict:
        return Verdict.BAD

    minimal = minimize(bad, verdict)
    assert len(minimal.steps) == 1


def test_minimize_preserves_final_output() -> None:
    bad = Trace(
        steps=(
            ToolStep(index=0, tool="f", args={}, output="x"),
            LlmStep(index=1, role="assistant", content="y"),
        ),
        final_output="THE-FINAL",
    )

    def verdict(t: Trace) -> Verdict:
        return Verdict.BAD

    minimal = minimize(bad, verdict)
    assert minimal.final_output == "THE-FINAL"
