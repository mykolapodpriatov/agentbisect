"""Tests for the step-aligned behavioral diff."""

from __future__ import annotations

from agentbisect.diff import diff
from agentbisect.types import LlmStep, ToolStep, Trace


def _trace(contents: list[str], final: str) -> Trace:
    steps = tuple(LlmStep(index=i, role="assistant", content=c) for i, c in enumerate(contents))
    return Trace(steps=steps, final_output=final)


def test_identical_traces_empty_diff() -> None:
    t = _trace(["a", "b", "c"], "same")
    d = diff(t, t)
    assert d.is_empty
    assert d.first_divergence is None
    assert not d.final_output_changed


def test_detects_first_divergence_point() -> None:
    left = _trace(["a", "b", "c"], "x")
    right = _trace(["a", "B-CHANGED", "c"], "x")
    d = diff(left, right)
    assert d.first_divergence == 1
    assert not d.is_empty
    diverging = [s for s in d.steps if not s.same]
    assert len(diverging) == 1
    assert diverging[0].index == 1


def test_detects_final_output_change() -> None:
    left = _trace(["a"], "good answer")
    right = _trace(["a"], "bad answer")
    d = diff(left, right)
    assert d.final_output_changed
    assert d.first_divergence is None  # steps match; only the final output differs
    assert not d.is_empty


def test_length_mismatch_marks_divergence() -> None:
    left = _trace(["a", "b"], "x")
    right = _trace(["a"], "x")
    d = diff(left, right)
    assert d.first_divergence == 1
    # The missing step is rendered explicitly.
    assert any("<missing>" in s.right for s in d.steps)


def test_tool_step_difference_detected() -> None:
    left = Trace(
        steps=(ToolStep(index=0, tool="search", args={"q": "a"}, output="r1"),),
        final_output="x",
    )
    right = Trace(
        steps=(ToolStep(index=0, tool="search", args={"q": "a"}, output="r2"),),
        final_output="x",
    )
    d = diff(left, right)
    assert d.first_divergence == 0
