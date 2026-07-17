"""Render a first-bad-change report: culprit + axis + minimal repro + behavioral diff.

Three renderers are provided, all consuming a
:class:`~agentbisect.driver.BisectionOutcome`:

* :func:`render_markdown` -- stable, test-friendly Markdown (the CLI's ``--markdown``).
* :func:`render_rich` -- a pretty terminal panel (the CLI's default).
* :func:`render_json` -- a stable, sorted, machine-readable JSON document (the CLI's
  ``--json``) carrying exactly the facts :func:`render_markdown` reports, so the two
  renderers never drift.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .diff import BehavioralDiff
from .driver import BisectionOutcome
from .types import BisectResult, Trace

__all__ = ["render_json", "render_markdown", "render_rich"]


def _diff_lines(bdiff: BehavioralDiff) -> list[str]:
    lines: list[str] = []
    if bdiff.is_empty:
        lines.append("- (no behavioral difference detected)")
        return lines
    if bdiff.first_divergence is not None:
        lines.append(f"- First diverging step: index {bdiff.first_divergence}")
    if bdiff.final_output_changed:
        lines.append(f"- Final output (last-good): {bdiff.left_final!r}")
        lines.append(f"- Final output (first-bad): {bdiff.right_final!r}")
    for sd in bdiff.steps:
        if not sd.same:
            lines.append(f"  - step {sd.index}: {sd.left}  =>  {sd.right}")
    return lines


def _repro_lines(repro: Trace) -> list[str]:
    lines = [f"- {len(repro.steps)} step(s); final: {repro.final_output!r}"]
    for step in repro.steps:
        lines.append(f"  - [{step.index}] {step.kind}")
    return lines


def render_markdown(outcome: BisectionOutcome) -> str:
    """Render the bisection outcome as deterministic Markdown."""
    result: BisectResult = outcome.result
    out: list[str] = ["# agentbisect report", ""]
    out.append(f"**Axis:** {result.axis}")
    out.append(f"**Probes:** {result.probes}")
    if outcome.used_passthrough:
        out.append(
            "**Note:** at least one verdict involved LIVE tool execution (passthrough); "
            "results are not fully hermetic."
        )
    out.append("")

    if result.first_bad is not None:
        out.append(f"## First bad change: `{result.first_bad.ref}`")
        if result.last_good is not None:
            out.append(f"- Last good: `{result.last_good.ref}`")
        out.append("")
        if outcome.artifacts_unavailable is not None:
            out.append("### Artifacts")
            out.append(f"- {outcome.artifacts_unavailable}")
            out.append("")
        if outcome.behavioral_diff is not None:
            out.append("### Behavioral diff (last-good vs first-bad)")
            out.extend(_diff_lines(outcome.behavioral_diff))
            out.append("")
        if outcome.minimal_repro is not None:
            out.append("### Minimal reproducing trace")
            out.extend(_repro_lines(outcome.minimal_repro))
            out.append("")
    else:
        out.append("## Result: ambiguous range (no single first-bad change)")
        if result.ambiguous_range is not None:
            lo, hi = result.ambiguous_range
            out.append(f"- The breaking change lies between `{lo.ref}` and `{hi.ref}`.")
        out.append(
            "- This happens when candidates at the boundary were untestable (skipped); "
            "narrow the range or resolve the skips to isolate a single culprit."
        )
        out.append("")

    out.append("## Candidates tested")
    for cand, verdict in result.steps_tested:
        out.append(f"- order {cand.order} `{cand.ref}` -> {verdict.value}")
    return "\n".join(out) + "\n"


def _diff_json(bdiff: BehavioralDiff | None) -> dict[str, Any] | None:
    """Serialize a behavioral diff, mirroring exactly what :func:`_diff_lines` reports."""
    if bdiff is None:
        return None
    return {
        "is_empty": bdiff.is_empty,
        "first_divergence": bdiff.first_divergence,
        "final_output_changed": bdiff.final_output_changed,
        "left_final": bdiff.left_final,
        "right_final": bdiff.right_final,
        # Only the diverging steps, matching the Markdown renderer's `if not same` filter.
        "differing_steps": [
            {"index": sd.index, "left": sd.left, "right": sd.right}
            for sd in bdiff.steps
            if not sd.same
        ],
    }


def _repro_json(repro: Trace | None) -> dict[str, Any] | None:
    """Serialize a minimal repro trace, mirroring exactly what :func:`_repro_lines` reports."""
    if repro is None:
        return None
    return {
        "num_steps": len(repro.steps),
        "final_output": repro.final_output,
        "steps": [{"index": step.index, "kind": step.kind} for step in repro.steps],
    }


def render_json(outcome: BisectionOutcome) -> str:
    """Render the bisection outcome as a stable, sorted, machine-readable JSON document.

    The schema carries exactly the facts :func:`render_markdown` reports -- axis, probes,
    the passthrough/artifact-availability flags, the first-bad/last-good refs (or ``null``),
    the ambiguous range, every tested step's order/ref/verdict, and the behavioral diff and
    minimal repro -- so the JSON and Markdown renderers never drift. Keys are sorted, so the
    output is byte-stable for a given outcome.
    """
    result: BisectResult = outcome.result
    payload: dict[str, Any] = {
        "axis": result.axis,
        "probes": result.probes,
        "used_passthrough": outcome.used_passthrough,
        "artifacts_unavailable": outcome.artifacts_unavailable,
        "first_bad": result.first_bad.ref if result.first_bad is not None else None,
        "last_good": result.last_good.ref if result.last_good is not None else None,
        "ambiguous_range": (
            [result.ambiguous_range[0].ref, result.ambiguous_range[1].ref]
            if result.ambiguous_range is not None
            else None
        ),
        "steps_tested": [
            {"order": cand.order, "ref": cand.ref, "verdict": verdict.value}
            for cand, verdict in result.steps_tested
        ],
        "behavioral_diff": _diff_json(outcome.behavioral_diff),
        "minimal_repro": _repro_json(outcome.minimal_repro),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_rich(outcome: BisectionOutcome, console: Console | None = None) -> None:
    """Print a pretty terminal report for the bisection outcome."""
    console = console or Console()
    result = outcome.result

    if result.first_bad is not None:
        title = f"[bold red]First bad change:[/] {result.first_bad.ref}"
        body = f"Axis: {result.axis}\nLast good: "
        body += result.last_good.ref if result.last_good else "(none)"
    else:
        title = "[bold yellow]Ambiguous range[/]"
        if result.ambiguous_range is not None:
            lo, hi = result.ambiguous_range
            body = f"Axis: {result.axis}\nBetween {lo.ref} and {hi.ref}"
        else:  # pragma: no cover - defensive
            body = f"Axis: {result.axis}"
    console.print(Panel(body, title=title))

    table = Table("order", "ref", "verdict")
    for cand, verdict in result.steps_tested:
        table.add_row(str(cand.order), cand.ref, verdict.value)
    console.print(table)

    if outcome.used_passthrough:
        console.print(
            "[yellow]Note:[/] at least one verdict involved live tool execution "
            "(passthrough); results are not fully hermetic."
        )
