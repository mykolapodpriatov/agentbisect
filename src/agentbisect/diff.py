"""Step-aligned behavioral diff between two traces.

``diff`` aligns two traces by step index and reports the first point where they diverge
plus a per-step side-by-side breakdown. Identical traces produce an empty diff.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .types import LlmStep, Step, ToolStep, Trace

__all__ = ["BehavioralDiff", "StepDiff", "diff"]


class StepDiff(BaseModel):
    """A single aligned step comparison."""

    model_config = ConfigDict(frozen=True)

    index: int
    same: bool
    left: str
    right: str


class BehavioralDiff(BaseModel):
    """The full behavioral comparison of two traces."""

    model_config = ConfigDict(frozen=True)

    #: Index of the first diverging step, or ``None`` if the step sequences match.
    first_divergence: int | None
    final_output_changed: bool
    left_final: str
    right_final: str
    steps: tuple[StepDiff, ...]

    @property
    def is_empty(self) -> bool:
        """True when the traces are behaviorally identical."""
        return self.first_divergence is None and not self.final_output_changed


def _summarize(step: Step | None) -> str:
    """Render a step to a compact, comparable one-liner."""
    if step is None:
        return "<missing>"
    if isinstance(step, LlmStep):
        calls = f" calls={list(step.tool_calls)}" if step.tool_calls else ""
        return f"llm[{step.role}]: {step.content}{calls}"
    if isinstance(step, ToolStep):
        return f"tool {step.tool}(args={step.args}) -> {step.output!r} ok={step.ok}"
    return repr(step)  # pragma: no cover - exhaustive above


def _same_step(a: Step | None, b: Step | None) -> bool:
    """Structural equality on the comparable fields of two steps."""
    return _summarize(a) == _summarize(b)


def diff(left: Trace, right: Trace) -> BehavioralDiff:
    """Return a step-aligned behavioral diff of ``left`` vs ``right``.

    Steps are aligned by position; the first index where they differ (or where one trace
    is shorter) is the divergence point.
    """
    n = max(len(left.steps), len(right.steps))
    step_diffs: list[StepDiff] = []
    first_divergence: int | None = None

    for i in range(n):
        ls = left.steps[i] if i < len(left.steps) else None
        rs = right.steps[i] if i < len(right.steps) else None
        same = _same_step(ls, rs)
        if not same and first_divergence is None:
            first_divergence = i
        step_diffs.append(StepDiff(index=i, same=same, left=_summarize(ls), right=_summarize(rs)))

    return BehavioralDiff(
        first_divergence=first_divergence,
        final_output_changed=left.final_output != right.final_output,
        left_final=left.final_output,
        right_final=right.final_output,
        steps=tuple(step_diffs),
    )
