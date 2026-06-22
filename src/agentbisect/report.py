"""Render a first-bad-change report: culprit + axis + minimal repro + behavioral diff.

Two renderers are provided: :func:`render_markdown` (stable, test-friendly, used by the
CLI's plain output) and :func:`render_rich` (a pretty terminal panel). Both consume a
:class:`~agentbisect.driver.BisectionOutcome`.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .diff import BehavioralDiff
from .driver import BisectionOutcome
from .types import BisectResult, Trace

__all__ = ["render_markdown", "render_rich"]


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
