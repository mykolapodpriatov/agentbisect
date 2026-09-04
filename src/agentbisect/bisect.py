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
  An optional ``max_probes`` cap aborts as an ambiguous range rather than guessing.
* **Determinism under ``workers > 1``.** Probes may be dispatched concurrently, but
  verdicts are folded into the memo in the deterministic index order, so the chosen
  candidate, the flaky candidate that is detected, and ``steps_tested`` never depend
  on which thread finished first. Concurrency buys wall-clock, never a different
  answer -- at the cost of extra probes, since a wave cannot stop early.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

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


class _ProbeCapReached(Exception):
    """Internal: ``max_probes`` budget exhausted before the next ``verdict_fn`` call."""


class _Memo:
    """Caches verdicts and detects flakiness (a candidate flipping between probes)."""

    def __init__(self, verdict_fn: VerdictFn, max_probes: int | None = None) -> None:
        self._fn = verdict_fn
        self._max_probes = max_probes
        self._cache: dict[int, Verdict] = {}
        self.order: list[tuple[Candidate, Verdict]] = []
        self.probes = 0

    def get(self, candidates: Sequence[Candidate], idx: int) -> Verdict:
        """Return the verdict for ``candidates[idx]``, detecting flaky re-probes."""
        if self._max_probes is not None and self.probes >= self._max_probes:
            raise _ProbeCapReached
        return self._record(candidates, idx, self._fn(candidates[idx]))

    def get_many(
        self,
        candidates: Sequence[Candidate],
        indices: Sequence[int],
        workers: int,
    ) -> dict[int, Verdict]:
        """Probe ``indices`` concurrently, returning ``{index: verdict}``.

        Verdicts are folded into the memo strictly in the order ``indices`` were
        given, never in completion order, so ``steps_tested``, the probe count
        and flaky detection stay identical to a sequential run. The returned map
        may be SHORTER than ``indices`` when ``max_probes`` cuts the batch off:
        the prefix that fits the budget is probed, which is exactly the prefix a
        sequential run would have reached.

        A ``verdict_fn`` that raises propagates from the first failing index in
        order, again so the error does not depend on thread scheduling.
        """
        budgeted = list(indices)
        if self._max_probes is not None:
            remaining = self._max_probes - self.probes
            if remaining <= 0:
                raise _ProbeCapReached
            budgeted = budgeted[:remaining]
        if not budgeted:
            return {}
        if workers <= 1 or len(budgeted) == 1:
            return {idx: self.get(candidates, idx) for idx in budgeted}

        with ThreadPoolExecutor(max_workers=min(workers, len(budgeted))) as pool:
            futures = {idx: pool.submit(self._fn, candidates[idx]) for idx in budgeted}
            return {idx: self._record(candidates, idx, futures[idx].result()) for idx in budgeted}

    def _record(self, candidates: Sequence[Candidate], idx: int, fresh: Verdict) -> Verdict:
        """Book one verdict: count it, check flakiness, append to the trail."""
        candidate = candidates[idx]
        self.probes += 1
        if idx in self._cache and self._cache[idx] != fresh:
            raise NonMonotonicError(
                f"candidate at order {candidate.order} (ref {candidate.ref!r}) returned "
                f"{self._cache[idx].value!r} then {fresh.value!r}: verdict is flaky / "
                "non-deterministic and cannot be bisected reliably"
            )
        self._cache[idx] = fresh
        self.order.append((candidate, fresh))
        return fresh


def bisect(
    candidates: Sequence[Candidate],
    verdict_fn: VerdictFn,
    *,
    max_probes: int | None = None,
    workers: int = 1,
) -> BisectResult:
    """Binary-search ``candidates`` (ordered old->new) for the first bad change.

    Parameters
    ----------
    candidates:
        The ordered candidate list (index 0 = oldest/expected-good endpoint, index -1 =
        newest/expected-bad endpoint). Must contain at least two candidates.
    verdict_fn:
        Maps a candidate to ``good``/``bad``/``skip``. The quarantine rule (diverged or
        nearest-substituted replays -> ``skip``) is expected to be folded in here.
    max_probes:
        Optional hard cap on ``verdict_fn`` calls, including the two endpoint probes.
        ``None`` means no cap. Hitting the cap returns the current bracket as an
        ambiguous range (``first_bad=None``) rather than a guessed culprit.
    workers:
        How many ``verdict_fn`` calls may be in flight at once. ``1`` (the default)
        is the sequential search. Above 1, the two endpoint probes run together and
        the skip fan-out is probed in waves.

        The result is unchanged: verdicts are folded in deterministic index order,
        and a wave still resolves to the first non-skip *in that order*, not the
        first to return. What does change is cost. A sequential fan-out stops at
        the first answer; a wave probes the whole wave, so ``probes`` and
        ``steps_tested`` grow. On a paid LLM judge that is real money, which is why
        this is opt-in.

        ``verdict_fn`` must be safe to call from several threads at once. The
        default one built by :func:`agentbisect.driver.make_verdict_fn` calls the
        project's ``AgentRunner``, and nothing in that contract promises thread
        safety, so leave this at 1 unless the runner is known to be reentrant.

    Returns
    -------
    BisectResult
        With ``first_bad`` set only for an adjacent good->bad transition, otherwise an
        ambiguous range with ``first_bad=None``.

    Raises
    ------
    ValueError
        If fewer than two candidates are supplied, ``max_probes`` is set and ``< 2``,
        or ``workers`` is below 1.
    UntestableEndpointError
        If an endpoint resolves ``skip``.
    NonMonotonicError
        If the first endpoint is ``bad``, the last is ``good``, or any candidate is flaky.
    """
    n = len(candidates)
    if n < 2:
        raise ValueError("bisect requires at least two candidates")
    if max_probes is not None and max_probes < 2:
        raise ValueError("max_probes must be at least 2 (both endpoints must be probed)")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    memo = _Memo(verdict_fn, max_probes=max_probes)

    # --- Endpoint validation (explicit first step) -------------------------------
    # The two endpoints are independent, so with workers > 1 they go out together
    # rather than costing two serial round trips on every run.
    if workers > 1:
        endpoints = memo.get_many(candidates, (0, n - 1), workers)
        if len(endpoints) < 2:
            raise _ProbeCapReached
        lo_verdict = endpoints[0]
        hi_verdict = endpoints[n - 1]
    else:
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
    try:
        while hi > lo + 1:
            mid = (lo + hi) // 2
            verdict, resolved = _probe_with_skip(candidates, memo, mid, lo, hi, workers=workers)
            if verdict is None:
                # The entire open interval (lo, hi) is skip -> ambiguous range.
                break
            assert resolved is not None
            if verdict is Verdict.GOOD:
                lo = resolved
            else:  # Verdict.BAD
                hi = resolved
    except _ProbeCapReached:
        return _build_result(
            candidates,
            memo,
            lo,
            hi,
            stop_reason=(
                f"stopped after {memo.probes} probes (max-probes={max_probes}); "
                "no single first-bad isolated"
            ),
        )

    return _build_result(candidates, memo, lo, hi)


def _outward_order(mid: int, lo: int, hi: int) -> list[int]:
    """The probe order for one search step: ``mid``, then out symmetrically.

    ``mid - 1``, ``mid + 1``, ``mid - 2``, ... staying strictly inside the open
    interval ``(lo, hi)``. Materialising the order (rather than walking it) is
    what lets a wave be dispatched concurrently while the *choice* among its
    results stays this exact, deterministic sequence.
    """
    order = [mid]
    offset = 1
    while True:
        left = mid - offset
        right = mid + offset
        left_ok = left > lo
        right_ok = right < hi
        if not left_ok and not right_ok:
            return order
        if left_ok:
            order.append(left)
        if right_ok:
            order.append(right)
        offset += 1


def _probe_with_skip(
    candidates: Sequence[Candidate],
    memo: _Memo,
    mid: int,
    lo: int,
    hi: int,
    *,
    workers: int = 1,
) -> tuple[Verdict | None, int | None]:
    """Probe ``mid``; on ``skip`` fan out (mid-1, mid+1, mid-2, ...) within ``(lo, hi)``.

    Returns ``(verdict, index)`` for the first non-skip candidate in that order, or
    ``(None, None)`` when every index in the open interval ``(lo, hi)`` is skip.

    With ``workers > 1`` the order is probed in waves instead of one at a time,
    and the winner is still the first non-skip *in the order*, not the first to
    return. A wave cannot stop early, so it costs probes a sequential walk would
    have saved.
    """
    order = _outward_order(mid, lo, hi)

    if workers <= 1:
        for idx in order:
            verdict = memo.get(candidates, idx)
            if verdict is not Verdict.SKIP:
                return verdict, idx
        return None, None

    for start in range(0, len(order), workers):
        wave = order[start : start + workers]
        verdicts = memo.get_many(candidates, wave, workers)
        for idx in wave:
            if idx not in verdicts:
                # The probe cap cut the wave short; nothing further was probed.
                return None, None
            if verdicts[idx] is not Verdict.SKIP:
                return verdicts[idx], idx
    return None, None


def _build_result(
    candidates: Sequence[Candidate],
    memo: _Memo,
    lo: int,
    hi: int,
    *,
    stop_reason: str | None = None,
) -> BisectResult:
    """Assemble the final result, emitting a single first_bad only when ``hi == lo+1``.

    A ``stop_reason`` (probe-cap) always yields ``first_bad=None`` even if the
    remaining bracket happens to look adjacent -- the cap means we did not finish
    confirming the boundary.
    """
    axis = candidates[0].axis
    last_good = candidates[lo]
    if stop_reason is None and hi == lo + 1:
        return BisectResult(
            axis=axis,
            first_bad=candidates[hi],
            last_good=last_good,
            ambiguous_range=None,
            steps_tested=tuple(memo.order),
            probes=memo.probes,
        )
    # Undetermined boundary (skip at the edge / all-skip interval / probe cap).
    return BisectResult(
        axis=axis,
        first_bad=None,
        last_good=last_good,
        ambiguous_range=(candidates[lo], candidates[hi]),
        steps_tested=tuple(memo.order),
        probes=memo.probes,
        stop_reason=stop_reason,
    )
