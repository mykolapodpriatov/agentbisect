"""Render a first-bad-change report: culprit + axis + minimal repro + behavioral diff.

Four renderers are provided, all consuming a
:class:`~agentbisect.driver.BisectionOutcome`:

* :func:`render_markdown` -- stable, test-friendly Markdown (the CLI's ``--markdown``).
* :func:`render_rich` -- a pretty terminal panel (the CLI's default).
* :func:`render_json` -- a stable, sorted, machine-readable JSON document (the CLI's
  ``--json``) carrying exactly the facts :func:`render_markdown` reports, so the two
  renderers never drift.
* :func:`render_html` -- a self-contained static HTML document (the CLI's ``--html``)
  carrying the same facts as a single shareable artifact with no external assets.
"""

from __future__ import annotations

import html
import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .diff import BehavioralDiff
from .driver import BisectionOutcome
from .types import BisectResult, Trace

__all__ = [
    "render_diff_json",
    "render_diff_markdown",
    "render_html",
    "render_json",
    "render_markdown",
    "render_rich",
]


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


def _diff_dict(bdiff: BehavioralDiff) -> dict[str, Any]:
    """Serialize a behavioral diff to a plain dict, mirroring :func:`_diff_lines`.

    Shared by the embedded outcome serializer (:func:`_diff_json`) and the standalone
    diff-subcommand serializer (:func:`render_diff_json`) so their shapes never drift.
    """
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


def _diff_json(bdiff: BehavioralDiff | None) -> dict[str, Any] | None:
    """Serialize a behavioral diff for embedding in the outcome-level JSON, or ``None``."""
    if bdiff is None:
        return None
    return _diff_dict(bdiff)


def render_diff_json(bdiff: BehavioralDiff) -> str:
    """Render a standalone behavioral diff as a stable, sorted JSON document.

    Consumed by the CLI's ``diff --json``. The schema matches the ``behavioral_diff``
    object embedded in :func:`render_json` (both go through :func:`_diff_dict`), so a
    standalone diff and a bisection's embedded diff are byte-for-byte the same shape.
    """
    return json.dumps(_diff_dict(bdiff), indent=2, sort_keys=True)


def render_diff_markdown(bdiff: BehavioralDiff) -> str:
    """Render a standalone behavioral diff (left vs right) as deterministic Markdown.

    Consumed by the CLI's ``diff --markdown``. Carries exactly the facts the human
    default of the ``diff`` command prints -- the empty-diff notice, the first-divergence
    index, each differing step, and any final-output change -- so the two never drift.
    """
    out: list[str] = ["# behavioral diff", ""]
    if bdiff.is_empty:
        out.append("- no behavioral difference")
        return "\n".join(out) + "\n"
    if bdiff.first_divergence is not None:
        out.append(f"- First divergence at step: {bdiff.first_divergence}")
    if bdiff.final_output_changed:
        out.append(f"- Final output (left): {bdiff.left_final!r}")
        out.append(f"- Final output (right): {bdiff.right_final!r}")
    differing = [sd for sd in bdiff.steps if not sd.same]
    if differing:
        out.append("")
        out.append("## Differing steps")
        for sd in differing:
            out.append(f"- step {sd.index}: {sd.left}  =>  {sd.right}")
    return "\n".join(out) + "\n"


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


#: Document scaffold for :func:`render_html`. Self-contained: the stylesheet is inlined,
#: so the rendered report is a single shareable file with no external assets.
_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentbisect report</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }
  h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
  code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px;
         font-family: ui-monospace, monospace; }
  table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
  th, td { border: 1px solid #ddd; padding: 0.35rem 0.6rem; text-align: left;
           vertical-align: top; }
  th { background: #f4f4f4; }
  .note { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 0.5rem 0.8rem; }
  .verdict-good { color: #2e7d32; }
  .verdict-bad { color: #c62828; }
  .verdict-skip { color: #757575; }
</style>
</head>
<body>
<h1>agentbisect report</h1>
"""

_HTML_TAIL = "</body>\n</html>\n"


def _diff_html(bdiff: BehavioralDiff) -> str:
    """Render a behavioral diff as an HTML fragment, mirroring :func:`_diff_lines`."""
    if bdiff.is_empty:
        return "<p>(no behavioral difference detected)</p>"
    items: list[str] = []
    if bdiff.first_divergence is not None:
        items.append(f"<li>First diverging step: index {bdiff.first_divergence}</li>")
    if bdiff.final_output_changed:
        items.append(
            f"<li>Final output (last-good): <code>{html.escape(repr(bdiff.left_final))}</code></li>"
        )
        items.append(
            "<li>Final output (first-bad): "
            f"<code>{html.escape(repr(bdiff.right_final))}</code></li>"
        )
    parts: list[str] = []
    if items:
        parts.append("<ul>")
        parts.extend(items)
        parts.append("</ul>")
    differing = [sd for sd in bdiff.steps if not sd.same]
    if differing:
        parts.append("<table>")
        parts.append("<thead><tr><th>step</th><th>last-good</th><th>first-bad</th></tr></thead>")
        parts.append("<tbody>")
        for sd in differing:
            parts.append(
                f"<tr><td>{sd.index}</td>"
                f"<td><code>{html.escape(sd.left)}</code></td>"
                f"<td><code>{html.escape(sd.right)}</code></td></tr>"
            )
        parts.append("</tbody></table>")
    return "\n".join(parts)


def _repro_html(repro: Trace) -> str:
    """Render a minimal repro trace as an HTML fragment, mirroring :func:`_repro_lines`."""
    parts = [
        f"<p>{len(repro.steps)} step(s); final: "
        f"<code>{html.escape(repr(repro.final_output))}</code></p>"
    ]
    if repro.steps:
        parts.append("<ul>")
        for step in repro.steps:
            parts.append(f"<li>[{step.index}] {html.escape(step.kind)}</li>")
        parts.append("</ul>")
    return "\n".join(parts)


def render_html(outcome: BisectionOutcome) -> str:
    """Render the bisection outcome as a self-contained static HTML document.

    Carries exactly the facts :func:`render_markdown` reports -- axis, probes, the
    passthrough/artifact-availability flags, the first-bad/last-good refs (or the
    ambiguous range), the behavioral diff, the minimal repro, and every tested candidate
    -- as a single shareable file with no external assets, so the HTML and Markdown
    renderers never drift. Every interpolated value is escaped with :func:`html.escape`.
    """
    result: BisectResult = outcome.result
    body: list[str] = [
        f"<p><strong>Axis:</strong> {html.escape(result.axis)}</p>",
        f"<p><strong>Probes:</strong> {result.probes}</p>",
    ]
    if outcome.used_passthrough:
        body.append(
            '<p class="note"><strong>Note:</strong> at least one verdict involved LIVE '
            "tool execution (passthrough); results are not fully hermetic.</p>"
        )

    if result.first_bad is not None:
        body.append(f"<h2>First bad change: <code>{html.escape(result.first_bad.ref)}</code></h2>")
        if result.last_good is not None:
            body.append(f"<p>Last good: <code>{html.escape(result.last_good.ref)}</code></p>")
        if outcome.artifacts_unavailable is not None:
            body.append("<h3>Artifacts</h3>")
            body.append(f"<p>{html.escape(outcome.artifacts_unavailable)}</p>")
        if outcome.behavioral_diff is not None:
            body.append("<h3>Behavioral diff (last-good vs first-bad)</h3>")
            body.append(_diff_html(outcome.behavioral_diff))
        if outcome.minimal_repro is not None:
            body.append("<h3>Minimal reproducing trace</h3>")
            body.append(_repro_html(outcome.minimal_repro))
    else:
        body.append("<h2>Result: ambiguous range (no single first-bad change)</h2>")
        if result.ambiguous_range is not None:
            lo, hi = result.ambiguous_range
            body.append(
                "<p>The breaking change lies between "
                f"<code>{html.escape(lo.ref)}</code> and "
                f"<code>{html.escape(hi.ref)}</code>.</p>"
            )
        body.append(
            "<p>This happens when candidates at the boundary were untestable (skipped); "
            "narrow the range or resolve the skips to isolate a single culprit.</p>"
        )

    body.append("<h2>Candidates tested</h2>")
    body.append("<table>")
    body.append("<thead><tr><th>order</th><th>ref</th><th>verdict</th></tr></thead>")
    body.append("<tbody>")
    for cand, verdict in result.steps_tested:
        verdict_class = f"verdict-{html.escape(verdict.value)}"
        body.append(
            f"<tr><td>{cand.order}</td>"
            f"<td><code>{html.escape(cand.ref)}</code></td>"
            f'<td class="{verdict_class}">{html.escape(verdict.value)}</td></tr>'
        )
    body.append("</tbody></table>")

    return _HTML_HEAD + "\n".join(body) + "\n" + _HTML_TAIL


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
