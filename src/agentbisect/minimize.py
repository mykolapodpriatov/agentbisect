"""Reduce a bad trace to a minimal reproducing prefix/subset.

``minimize`` greedily drops trailing steps while a verdict function still reports
``bad``, yielding the shortest still-failing prefix. The pass count is bounded (one pass
per step at most) so it always terminates; the indices on the surviving steps are
re-numbered to stay contiguous.
"""

from __future__ import annotations

from collections.abc import Callable

from .types import Trace, Verdict

__all__ = ["VerdictFn", "minimize"]

#: A verdict function maps a (candidate) trace to a verdict. For minimization only the
#: BAD / not-BAD distinction matters.
VerdictFn = Callable[[Trace], Verdict]


def _reindex(trace: Trace) -> Trace:
    """Return ``trace`` with steps re-numbered 0..n-1 to keep indices contiguous."""
    new_steps = tuple(step.model_copy(update={"index": i}) for i, step in enumerate(trace.steps))
    return Trace(steps=new_steps, final_output=trace.final_output)


def minimize(
    bad_trace: Trace,
    verdict_fn: VerdictFn,
    *,
    max_passes: int | None = None,
) -> Trace:
    """Greedily shrink ``bad_trace`` to a minimal still-``bad`` prefix.

    Parameters
    ----------
    bad_trace:
        A trace the oracle judges ``bad``.
    verdict_fn:
        Maps a candidate trace to a verdict; minimization keeps shrinking while it
        stays ``bad``.
    max_passes:
        Optional cap on shrink attempts (defaults to one attempt per step). Guarantees
        termination.

    Returns
    -------
    Trace
        The shortest trailing-trimmed prefix that the verdict function still calls
        ``bad`` (indices re-numbered). If the input is not ``bad``, it is returned
        unchanged (re-indexed).
    """
    if verdict_fn(bad_trace) is not Verdict.BAD:
        return _reindex(bad_trace)

    steps = list(bad_trace.steps)
    budget = max_passes if max_passes is not None else len(steps)

    # Drop trailing steps one at a time while the result stays bad.
    while len(steps) > 1 and budget > 0:
        budget -= 1
        candidate = Trace(steps=tuple(steps[:-1]), final_output=bad_trace.final_output)
        if verdict_fn(candidate) is Verdict.BAD:
            steps = steps[:-1]
        else:
            break

    return _reindex(Trace(steps=tuple(steps), final_output=bad_trace.final_output))
