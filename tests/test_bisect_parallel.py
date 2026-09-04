"""Concurrency must buy wall-clock and nothing else.

Every test here compares ``workers > 1`` against the sequential search on the
same scripted verdicts: same culprit, same trail, same errors. The one thing
allowed to differ is the probe count, because a wave cannot stop early.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from agentbisect.bisect import NonMonotonicError, UntestableEndpointError, bisect
from agentbisect.types import AgentConfig, Candidate, Verdict

G = Verdict.GOOD
B = Verdict.BAD
S = Verdict.SKIP


def _candidates(n: int) -> list[Candidate]:
    base = AgentConfig(system_prompt="p", model="m")
    return [
        Candidate(axis="model", ref=f"v{i}", config=base.with_overrides(model=f"m{i}"), order=i)
        for i in range(n)
    ]


def _from_list(verdicts: list[Verdict]) -> Callable[[Candidate], Verdict]:
    def fn(candidate: Candidate) -> Verdict:
        return verdicts[candidate.order]

    return fn


def _summary(result: object) -> tuple[object, ...]:
    """The parts of a result that must not depend on concurrency."""
    first_bad = getattr(result, "first_bad", None)
    last_good = getattr(result, "last_good", None)
    ambiguous = getattr(result, "ambiguous_range", None)
    return (
        first_bad.order if first_bad is not None else None,
        last_good.order if last_good is not None else None,
        tuple(c.order for c in ambiguous) if ambiguous is not None else None,
        tuple((c.order, v) for c, v in result.steps_tested),  # type: ignore[attr-defined]
    )


# --------------------------------------------------------------- identical results


@pytest.mark.parametrize("workers", [2, 3, 8])
@pytest.mark.parametrize(
    "verdicts",
    [
        [G, G, G, B, B],
        [G, B, B, B, B],
        [G, G, G, G, B],
        [G, S, G, B, B],
        [G, S, S, S, B],
        [G, G, S, B, B, B, B],
        [G] * 8 + [B] * 8,
        [G, S, G, S, G, S, B, B],
    ],
)
def test_parallel_matches_sequential(verdicts: list[Verdict], workers: int) -> None:
    sequential = bisect(_candidates(len(verdicts)), _from_list(verdicts))
    parallel = bisect(_candidates(len(verdicts)), _from_list(verdicts), workers=workers)

    assert _summary(parallel)[:3] == _summary(sequential)[:3]


def test_parallel_matches_sequential_over_every_boundary() -> None:
    """Every single-transition list of length 10, both search modes."""
    n = 10
    for boundary in range(1, n):
        verdicts = [G] * boundary + [B] * (n - boundary)
        sequential = bisect(_candidates(n), _from_list(verdicts))
        parallel = bisect(_candidates(n), _from_list(verdicts), workers=4)
        assert parallel.first_bad is not None
        assert sequential.first_bad is not None
        assert parallel.first_bad.order == sequential.first_bad.order == boundary


# ------------------------------------------------------- order, not completion order


def test_trail_order_ignores_completion_order() -> None:
    """A verdict_fn whose later indices answer first still records in index order."""
    verdicts = [G, G, G, B, S, B, B, B, B]

    def fn(candidate: Candidate) -> Verdict:
        # Earlier candidates take longer, so completion order is the reverse of
        # index order inside every wave.
        time.sleep(0.01 * (len(verdicts) - candidate.order))
        return verdicts[candidate.order]

    sequential = bisect(_candidates(len(verdicts)), _from_list(verdicts))
    parallel = bisect(_candidates(len(verdicts)), fn, workers=4)

    assert parallel.first_bad is not None
    assert sequential.first_bad is not None
    assert parallel.first_bad.order == sequential.first_bad.order

    orders = [c.order for c, _ in parallel.steps_tested]
    # Endpoints, then the first wave in its outward order (mid, mid-1, mid+1,
    # mid-2), not in the order the threads happened to finish.
    assert orders[:2] == [0, 8]
    assert orders[2:6] == [4, 3, 5, 2]


def test_skip_run_resolves_to_the_same_index() -> None:
    """A wave picks the first non-skip in the outward order, not the first to return."""
    # mid=4 is a skip. The outward order is 3, 5, 2, so 3 must win even though 5
    # is also non-skip and answers sooner.
    verdicts = [G, G, G, B, S, B, B, B, B]

    def fn(candidate: Candidate) -> Verdict:
        if candidate.order == 3:
            time.sleep(0.05)  # the winner is also the slowest
        return verdicts[candidate.order]

    sequential = bisect(_candidates(len(verdicts)), _from_list(verdicts))
    parallel = bisect(_candidates(len(verdicts)), fn, workers=4)

    assert parallel.first_bad is not None
    assert sequential.first_bad is not None
    assert parallel.first_bad.order == sequential.first_bad.order
    orders = [c.order for c, _ in parallel.steps_tested]
    assert orders.index(3) < orders.index(5)


# ----------------------------------------------------------------- errors and caps


def test_flaky_candidate_still_raises() -> None:
    """Index 2 is re-probed after the bracket narrows, and flips. Both modes catch it."""
    seen: dict[int, int] = {}
    lock = threading.Lock()

    def fn(candidate: Candidate) -> Verdict:
        with lock:
            seen[candidate.order] = seen.get(candidate.order, 0) + 1
            count = seen[candidate.order]
        if candidate.order == 2:
            return S if count == 1 else B
        return G if candidate.order < 3 else B

    with pytest.raises(NonMonotonicError, match="flaky"):
        bisect(_candidates(5), fn, workers=1)

    seen.clear()
    with pytest.raises(NonMonotonicError, match="flaky"):
        bisect(_candidates(5), fn, workers=4)


def test_untestable_endpoint_still_raises_with_workers() -> None:
    verdicts = [S, G, G, B, B]
    with pytest.raises(UntestableEndpointError):
        bisect(_candidates(len(verdicts)), _from_list(verdicts), workers=4)


def test_non_monotonic_endpoints_still_raise_with_workers() -> None:
    with pytest.raises(NonMonotonicError):
        bisect(_candidates(5), _from_list([B, B, B, B, B]), workers=4)
    with pytest.raises(NonMonotonicError):
        bisect(_candidates(5), _from_list([G, G, G, G, G]), workers=4)


@pytest.mark.parametrize("workers", [1, 2, 4, 16])
@pytest.mark.parametrize("cap", [2, 3, 5, 9])
def test_max_probes_is_never_exceeded(cap: int, workers: int) -> None:
    verdicts = [G, S, S, S, S, S, S, S, B]
    result = bisect(
        _candidates(len(verdicts)), _from_list(verdicts), max_probes=cap, workers=workers
    )
    assert result.probes <= cap


def test_workers_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="workers"):
        bisect(_candidates(4), _from_list([G, G, B, B]), workers=0)


def test_verdict_fn_error_propagates_from_the_first_index_in_order() -> None:
    """An exception must not depend on which thread got there first."""

    class Boom(RuntimeError):
        pass

    verdicts = [G, S, S, S, B]

    def fn(candidate: Candidate) -> Verdict:
        if candidate.order in (2, 3):
            if candidate.order == 3:
                time.sleep(0.05)
            raise Boom(f"order {candidate.order}")
        return verdicts[candidate.order]

    with pytest.raises(Boom, match="order 2"):
        bisect(_candidates(len(verdicts)), fn, workers=4)


# ------------------------------------------------------------------------ the point


def test_endpoints_are_probed_concurrently() -> None:
    """The two endpoint probes are independent; workers > 1 overlaps them."""
    started = threading.Barrier(2, timeout=5.0)

    def fn(candidate: Candidate) -> Verdict:
        if candidate.order in (0, 4):
            # Both endpoints must be in flight at once, or this deadlocks and
            # the barrier's timeout fails the test.
            started.wait()
        return G if candidate.order < 3 else B

    result = bisect(_candidates(5), fn, workers=2)
    assert result.first_bad is not None
    assert result.first_bad.order == 3
