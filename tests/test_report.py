"""Tests for report rendering (Markdown + rich)."""

from __future__ import annotations

from rich.console import Console

from agentbisect.diff import diff
from agentbisect.driver import BisectionOutcome
from agentbisect.report import render_markdown, render_rich
from agentbisect.types import AgentConfig, BisectResult, Candidate, LlmStep, Trace, Verdict


def _cand(order: int, ref: str) -> Candidate:
    base = AgentConfig(system_prompt="p", model="m")
    return Candidate(axis="prompt", ref=ref, order=order, config=base)


def _first_bad_outcome() -> BisectionOutcome:
    last_good = _cand(0, "good-sha")
    first_bad = _cand(1, "bad-sha")
    result = BisectResult(
        axis="prompt",
        first_bad=first_bad,
        last_good=last_good,
        steps_tested=((last_good, Verdict.GOOD), (first_bad, Verdict.BAD)),
        probes=2,
    )
    good_trace = Trace(steps=(LlmStep(index=0, role="a", content="x"),), final_output="good")
    bad_trace = Trace(steps=(LlmStep(index=0, role="a", content="y"),), final_output="bad")
    return BisectionOutcome(
        result,
        minimal_repro=bad_trace,
        behavioral_diff=diff(good_trace, bad_trace),
        used_passthrough=False,
    )


def _ambiguous_outcome() -> BisectionOutcome:
    lo = _cand(0, "lo-sha")
    hi = _cand(2, "hi-sha")
    result = BisectResult(
        axis="prompt",
        first_bad=None,
        last_good=lo,
        ambiguous_range=(lo, hi),
        steps_tested=((lo, Verdict.GOOD), (hi, Verdict.BAD)),
        probes=2,
    )
    return BisectionOutcome(result, minimal_repro=None, behavioral_diff=None, used_passthrough=True)


def test_markdown_first_bad() -> None:
    md = render_markdown(_first_bad_outcome())
    assert "First bad change: `bad-sha`" in md
    assert "Last good: `good-sha`" in md
    assert "Behavioral diff" in md
    assert "Minimal reproducing trace" in md


def test_markdown_ambiguous() -> None:
    md = render_markdown(_ambiguous_outcome())
    assert "ambiguous range" in md
    assert "between `lo-sha` and `hi-sha`" in md
    assert "LIVE tool execution" in md  # passthrough note surfaced


def test_rich_first_bad_renders() -> None:
    console = Console(record=True, width=100)
    render_rich(_first_bad_outcome(), console)
    text = console.export_text()
    assert "First bad change" in text
    assert "bad-sha" in text


def test_rich_ambiguous_renders() -> None:
    console = Console(record=True, width=100)
    render_rich(_ambiguous_outcome(), console)
    text = console.export_text()
    assert "Ambiguous range" in text
    assert "live tool execution" in text


def test_markdown_first_bad_with_empty_diff() -> None:
    # A first-bad result whose behavioral diff is empty exercises the empty-diff branch.
    last_good = _cand(0, "good-sha")
    first_bad = _cand(1, "bad-sha")
    result = BisectResult(
        axis="prompt",
        first_bad=first_bad,
        last_good=last_good,
        steps_tested=((last_good, Verdict.GOOD), (first_bad, Verdict.BAD)),
        probes=2,
    )
    same = Trace(steps=(LlmStep(index=0, role="a", content="x"),), final_output="same")
    outcome = BisectionOutcome(
        result,
        minimal_repro=same,
        behavioral_diff=diff(same, same),  # identical -> empty diff
        used_passthrough=False,
    )
    md = render_markdown(outcome)
    assert "no behavioral difference detected" in md
