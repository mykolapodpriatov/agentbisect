"""The pure binary-search core with git-bisect-compatible good/bad/skip semantics.

This module is intentionally a *pure function* of ``(candidates, verdict_fn)`` so it can
be exhaustively tested with synthetic ordered lists and a scripted oracle. The
verdict function is where the replay -> oracle -> quarantine pipeline is folded in
(see :func:`agentbisect.driver.make_verdict_fn`); the search itself knows nothing about
replay or oracles.

Guarantees:

* **Endpoint validation first.** Both endpoints are probed under the same rules. An
  endpoint that resolves ``skip`` raises :class:`UntestableEndpointError`; a first
  endpoint that is ``bad`` or a last endpoint that is ``good`` raises
  :class:`NonMonotonicError`. The two are kept distinct.
* **Single ``first_bad`` only for an adjacent transition.** A confident single culprit
  is returned *only* when the confirmed-good and confirmed-bad indices become adjacent
  (``hi == lo + 1``). Otherwise an *ambiguous range* with ``first_bad=None`` is returned.
* **Skip handling.** A ``skip`` at ``mid`` probes outward strictly within ``(lo, hi)``;
  an all-skip open interval terminates as an ambiguous range (no infinite loop).
* **Flaky detection.** A candidate that flips verdict between probes raises
  :class:`NonMonotonicError` -- never a confidently-wrong ``first_bad``.
* **Bounded probes.** Total probes are bounded; the function always terminates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .types import BisectResult, Candidate, Verdict

__all__ = [
    "BisectError",
    "NonMonotonicError",
    "UntestableEndpointError",
    "bisect",
]

VerdictFn = Callable[[Candidate], Verdict]


class BisectError(Exception):
    """Base class for bisect precondition/consistency failures."""


class UntestableEndpointError(BisectError):
    """Raised when an endpoint resolves ``skip`` (quarantined/undecidable/untestable).

    You cannot bisect without two trustworthy endpoints, so this is kept distinct from
    :class:`NonMonotonicError` -- it tells the user to fix the endpoints, not the
    monotonicity assumption.
    """


class NonMonotonicError(BisectError):
    """Raised when verdicts violate the single good->bad transition assumption.

    Triggered when the first endpoint is ``bad`` / the last endpoint is ``good``, or
    when a candidate flips verdict between probes (flaky).
    """


class _Memo:
    """Caches verdicts and detects flakiness (a candidate flipping between probes)."""

    def __init__(self, verdict_fn: VerdictFn) -> None:
        self._fn = verdict_fn
        self._cache: dict[int, Verdict] = {}
        self.order: list[tuple[Candidate, Verdict]] = []
        self.probes = 0

    def get(self, candidates: Sequence[Candidate], idx: int) -> Verdict:
        """Return the verdict for ``candidates[idx]``, detecting flaky re-probes."""
        candidate = candidates[idx]
        self.probes += 1
        fresh = self._fn(candidate)
        if idx in self._cache and self._cache[idx] != fresh:
            raise NonMonotonicError(
                f"candidate at order {candidate.order} (ref {candidate.ref!r}) returned "
                f"{self._cache[idx].value!r} then {fresh.value!r}: verdict is flaky / "
                "non-deterministic and cannot be bisected reliably"
            )
        self._cache[idx] = fresh
        self.order.append((candidate, fresh))
        return fresh


def bisect(candidates: Sequence[Candidate], verdict_fn: VerdictFn) -> BisectResult:
    """Binary-search ``candidates`` (ordered old->new) for the first bad change.

    Parameters
    ----------
    candidates:
        The ordered candidate list (index 0 = oldest/expected-good endpoint, index -1 =
        newest/expected-bad endpoint). Must contain at least two candidates.
    verdict_fn:
        Maps a candidate to ``good``/``bad``/``skip``. The quarantine rule (diverged or
        nearest-substituted replays -> ``skip``) is expected to be folded in here.

    Returns
    -------
    BisectResult
        With ``first_bad`` set only for an adjacent good->bad transition, otherwise an
        ambiguous range with ``first_bad=None``.

    Raises
    ------
    ValueError
        If fewer than two candidates are supplied.
    UntestableEndpointError
        If an endpoint resolves ``skip``.
    NonMonotonicError
        If the first endpoint is ``bad``, the last is ``good``, or any candidate is flaky.
    """
    n = len(candidates)
    if n < 2:
        raise ValueError("bisect requires at least two candidates")

    memo = _Memo(verdict_fn)

    # --- Endpoint validation (explicit first step) -------------------------------
    lo_verdict = memo.get(candidates, 0)
    hi_verdict = memo.get(candidates, n - 1)

    if lo_verdict is Verdict.SKIP or hi_verdict is Verdict.SKIP:
        # Check each endpoint independently so a both-skip case names *both* ends.
        untestable = [
            name
            for name, verdict in (("first", lo_verdict), ("last", hi_verdict))
            if verdict is Verdict.SKIP
        ]
        which = " and ".join(untestable)
        plural = "s" if len(untestable) > 1 else ""
        raise UntestableEndpointError(
            f"the {which} endpoint{plural} resolved 'skip' (quarantined or undecidable); "
            "cannot bisect without two trustworthy endpoints"
        )
    if lo_verdict is Verdict.BAD:
        raise NonMonotonicError(
            "the first endpoint is 'bad'; expected 'good' (the regression must be "
            "introduced somewhere after the oldest candidate)"
        )
    if hi_verdict is Verdict.GOOD:
        raise NonMonotonicError(
            "the last endpoint is 'good'; expected 'bad' (there is no regression to "
            "bisect across this range)"
        )

    lo = 0  # highest index confirmed good
    hi = n - 1  # lowest index confirmed bad

    # --- Binary search with skip-aware outward probing ---------------------------
    while hi > lo + 1:
        mid = (lo + hi) // 2
        verdict, resolved = _probe_with_skip(candidates, memo, mid, lo, hi)
        if verdict is None:
            # The entire open interval (lo, hi) is skip -> ambiguous range.
            break
        assert resolved is not None
        if verdict is Verdict.GOOD:
            lo = resolved
        else:  # Verdict.BAD
            hi = resolved

    return _build_result(candidates, memo, lo, hi)


def _probe_with_skip(
    candidates: Sequence[Candidate],
    memo: _Memo,
    mid: int,
    lo: int,
    hi: int,
) -> tuple[Verdict | None, int | None]:
    """Probe ``mid``; on ``skip`` fan out (mid-1, mid+1, mid-2, ...) within ``(lo, hi)``.

    Returns ``(verdict, index)`` for the first non-skip candidate found, or
    ``(None, None)`` when every index in the open interval ``(lo, hi)`` is skip.
    """
    verdict = memo.get(candidates, mid)
    if verdict is not Verdict.SKIP:
        return verdict, mid

    # Fan outward symmetrically, staying strictly inside (lo, hi).
    offset = 1
    while True:
        left = mid - offset
        right = mid + offset
        left_ok = left > lo
        right_ok = right < hi
        if not left_ok and not right_ok:
            return None, None
        if left_ok:
            v = memo.get(candidates, left)
            if v is not Verdict.SKIP:
                return v, left
        if right_ok:
            v = memo.get(candidates, right)
            if v is not Verdict.SKIP:
                return v, right
        offset += 1


def _build_result(
    candidates: Sequence[Candidate],
    memo: _Memo,
    lo: int,
    hi: int,
) -> BisectResult:
    """Assemble the final result, emitting a single first_bad only when ``hi == lo+1``."""
    axis = candidates[0].axis
    last_good = candidates[lo]
    if hi == lo + 1:
        return BisectResult(
            axis=axis,
            first_bad=candidates[hi],
            last_good=last_good,
            ambiguous_range=None,
            steps_tested=tuple(memo.order),
            probes=memo.probes,
        )
    # Undetermined boundary (skip at the edge / all-skip interval) -> ambiguous range.
    return BisectResult(
        axis=axis,
        first_bad=None,
        last_good=last_good,
        ambiguous_range=(candidates[lo], candidates[hi]),
        steps_tested=tuple(memo.order),
        probes=memo.probes,
    )
