"""Tests for report rendering (Markdown + rich + JSON)."""

from __future__ import annotations

import json

from rich.console import Console

from agentbisect.diff import diff
from agentbisect.driver import BisectionOutcome
from agentbisect.report import render_html, render_json, render_markdown, render_rich
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


def _empty_diff_outcome() -> BisectionOutcome:
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
    return BisectionOutcome(
        result,
        minimal_repro=same,
        behavioral_diff=diff(same, same),  # identical -> empty diff
        used_passthrough=False,
    )


def _passthrough_outcome() -> BisectionOutcome:
    # A first-bad result where a post-bisect artifact replay was non-reproducible, so the
    # artifacts are unavailable and passthrough was surfaced.
    result = _first_bad_outcome().result
    return BisectionOutcome(
        result,
        minimal_repro=None,
        behavioral_diff=None,
        used_passthrough=True,
        artifacts_unavailable="unavailable (replay diverged): artifacts not built",
    )


def test_json_first_bad_round_trips() -> None:
    data = json.loads(render_json(_first_bad_outcome()))
    assert data["axis"] == "prompt"
    assert data["probes"] == 2
    assert data["first_bad"] == "bad-sha"
    assert data["last_good"] == "good-sha"
    assert data["ambiguous_range"] is None
    assert data["used_passthrough"] is False
    assert data["artifacts_unavailable"] is None
    # steps_tested carries order/ref/verdict for each probe, in test order.
    assert data["steps_tested"] == [
        {"order": 0, "ref": "good-sha", "verdict": "good"},
        {"order": 1, "ref": "bad-sha", "verdict": "bad"},
    ]
    # The behavioral diff mirrors the Markdown report's facts.
    bdiff = data["behavioral_diff"]
    assert bdiff["is_empty"] is False
    assert bdiff["final_output_changed"] is True
    assert bdiff["left_final"] == "good"
    assert bdiff["right_final"] == "bad"
    assert bdiff["differing_steps"] and bdiff["differing_steps"][0]["index"] == 0
    # And so does the minimal repro.
    assert data["minimal_repro"] == {
        "num_steps": 1,
        "final_output": "bad",
        "steps": [{"index": 0, "kind": "llm"}],
    }


def test_json_ambiguous_round_trips() -> None:
    data = json.loads(render_json(_ambiguous_outcome()))
    assert data["first_bad"] is None
    assert data["last_good"] == "lo-sha"
    assert data["ambiguous_range"] == ["lo-sha", "hi-sha"]
    assert data["behavioral_diff"] is None
    assert data["minimal_repro"] is None


def test_json_empty_diff_round_trips() -> None:
    data = json.loads(render_json(_empty_diff_outcome()))
    assert data["behavioral_diff"]["is_empty"] is True
    assert data["behavioral_diff"]["first_divergence"] is None
    assert data["behavioral_diff"]["final_output_changed"] is False
    assert data["behavioral_diff"]["differing_steps"] == []


def test_json_passthrough_and_unavailable_artifacts_round_trip() -> None:
    data = json.loads(render_json(_passthrough_outcome()))
    assert data["used_passthrough"] is True
    assert data["artifacts_unavailable"].startswith("unavailable")
    assert data["first_bad"] == "bad-sha"
    assert data["behavioral_diff"] is None
    assert data["minimal_repro"] is None


def test_json_is_sorted_and_stable() -> None:
    # sort_keys makes the output byte-stable and independent of construction order.
    out = render_json(_first_bad_outcome())
    assert out == render_json(_first_bad_outcome())
    top_level_keys = list(json.loads(out).keys())
    assert top_level_keys == sorted(top_level_keys)


def test_html_first_bad_carries_markdown_facts() -> None:
    doc = render_html(_first_bad_outcome())
    # A self-contained document (inlined styles, no external assets).
    assert doc.startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")
    assert "<style>" in doc and "href=" not in doc and "<script" not in doc
    # The same facts the Markdown renderer carries.
    assert "First bad change" in doc
    assert "bad-sha" in doc
    assert "Last good" in doc and "good-sha" in doc
    assert "Behavioral diff" in doc
    assert "Minimal reproducing trace" in doc
    assert "Candidates tested" in doc
    # Verdicts are surfaced in the tested-candidates table.
    assert ">good<" in doc and ">bad<" in doc


def test_html_ambiguous_surfaces_range_and_passthrough() -> None:
    doc = render_html(_ambiguous_outcome())
    assert "ambiguous range" in doc
    assert "lo-sha" in doc and "hi-sha" in doc
    assert "LIVE tool execution" in doc  # passthrough note surfaced


def test_html_escapes_untrusted_refs() -> None:
    # A ref carrying HTML metacharacters must be escaped, never injected as raw markup.
    base = AgentConfig(system_prompt="p", model="m")
    hostile = Candidate(axis="prompt", ref="<script>alert(1)</script>&", order=1, config=base)
    last_good = _cand(0, "good-sha")
    result = BisectResult(
        axis="prompt",
        first_bad=hostile,
        last_good=last_good,
        steps_tested=((last_good, Verdict.GOOD), (hostile, Verdict.BAD)),
        probes=2,
    )
    outcome = BisectionOutcome(
        result, minimal_repro=None, behavioral_diff=None, used_passthrough=False
    )
    doc = render_html(outcome)
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;&amp;" in doc


def test_html_empty_diff_branch() -> None:
    doc = render_html(_empty_diff_outcome())
    assert "no behavioral difference detected" in doc


def test_html_is_stable() -> None:
    assert render_html(_first_bad_outcome()) == render_html(_first_bad_outcome())


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
