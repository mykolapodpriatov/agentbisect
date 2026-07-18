"""Tests for the step-aligned behavioral diff."""

from __future__ import annotations

import json

from agentbisect.diff import diff
from agentbisect.report import render_diff_json, render_diff_markdown
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


def test_render_diff_json_round_trips_differences() -> None:
    left = _trace(["a", "b"], "good")
    right = _trace(["a", "B-CHANGED"], "bad")
    data = json.loads(render_diff_json(diff(left, right)))
    assert data["is_empty"] is False
    assert data["first_divergence"] == 1
    assert data["final_output_changed"] is True
    assert data["left_final"] == "good"
    assert data["right_final"] == "bad"
    # Only the diverging steps are serialized.
    assert [s["index"] for s in data["differing_steps"]] == [1]


def test_render_diff_json_empty_diff() -> None:
    t = _trace(["a", "b"], "same")
    data = json.loads(render_diff_json(diff(t, t)))
    assert data["is_empty"] is True
    assert data["first_divergence"] is None
    assert data["final_output_changed"] is False
    assert data["differing_steps"] == []


def test_render_diff_json_is_sorted_and_stable() -> None:
    d = diff(_trace(["a"], "x"), _trace(["b"], "y"))
    out = render_diff_json(d)
    assert out == render_diff_json(d)
    assert list(json.loads(out).keys()) == sorted(json.loads(out).keys())


def test_render_diff_markdown_reports_facts() -> None:
    left = _trace(["a", "b"], "good")
    right = _trace(["a", "B-CHANGED"], "bad")
    md = render_diff_markdown(diff(left, right))
    assert md.startswith("# behavioral diff")
    assert "First divergence at step: 1" in md
    assert "## Differing steps" in md
    assert "step 1:" in md
    # Final-output change is carried with left/right framing.
    assert "Final output (left): 'good'" in md
    assert "Final output (right): 'bad'" in md


def test_render_diff_markdown_empty_diff() -> None:
    t = _trace(["a", "b"], "same")
    md = render_diff_markdown(diff(t, t))
    assert "no behavioral difference" in md
    assert "Differing steps" not in md
