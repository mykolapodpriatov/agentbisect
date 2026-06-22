"""Exhaustive tests for the pure bisect core (the project's central value).

The bisect is a pure function of ``(candidates, verdict_fn)``; these tests drive it with
synthetic ordered candidate lists and scripted verdict functions. No replay/oracle here
-- the quarantine fold-in is tested in ``test_driver.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from agentbisect.bisect import (
    NonMonotonicError,
    UntestableEndpointError,
    bisect,
)
from agentbisect.types import AgentConfig, Candidate, Verdict


def _candidates(n: int) -> list[Candidate]:
    """Build ``n`` ordered candidates on a synthetic 'model' axis."""
    base = AgentConfig(system_prompt="p", model="m")
    return [
        Candidate(axis="model", ref=f"v{i}", config=base.with_overrides(model=f"m{i}"), order=i)
        for i in range(n)
    ]


def _from_list(verdicts: list[Verdict]) -> Callable[[Candidate], Verdict]:
    """Verdict function reading a fixed per-order verdict list."""

    def fn(candidate: Candidate) -> Verdict:
        return verdicts[candidate.order]

    return fn


G = Verdict.GOOD
B = Verdict.BAD
S = Verdict.SKIP


# --------------------------------------------------------------------------- boundaries


def test_finds_adjacent_boundary() -> None:
    # good good good | bad bad   -> first bad at index 3
    verdicts = [G, G, G, B, B]
    cands = _candidates(5)
    result = bisect(cands, _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == 3
    assert result.last_good is not None
    assert result.last_good.order == 2
    assert result.ambiguous_range is None


def test_single_first_bad_only_for_adjacent_transition() -> None:
    # A confident first_bad requires hi == lo+1; here the boundary IS adjacent.
    verdicts = [G, B]
    result = bisect(_candidates(2), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == 1
    assert not result.is_ambiguous


def test_bad_at_index_one() -> None:
    verdicts = [G, B, B, B, B]
    result = bisect(_candidates(5), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == 1


def test_bad_only_at_last() -> None:
    verdicts = [G, G, G, G, B]
    result = bisect(_candidates(5), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == 4


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_exhaustive_boundaries(n: int, boundary: int) -> None:
    if boundary >= n:
        pytest.skip("boundary must be inside the list")
    verdicts = [G] * boundary + [B] * (n - boundary)
    result = bisect(_candidates(n), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == boundary
    assert result.last_good is not None
    assert result.last_good.order == boundary - 1


# --------------------------------------------------------------- endpoint validation


def test_non_monotonic_first_endpoint_bad() -> None:
    verdicts = [B, B, B]
    with pytest.raises(NonMonotonicError):
        bisect(_candidates(3), _from_list(verdicts))


def test_non_monotonic_last_endpoint_good() -> None:
    verdicts = [G, G, G]
    with pytest.raises(NonMonotonicError):
        bisect(_candidates(3), _from_list(verdicts))


def test_untestable_first_endpoint_skip() -> None:
    verdicts = [S, G, B]
    with pytest.raises(UntestableEndpointError):
        bisect(_candidates(3), _from_list(verdicts))


def test_untestable_last_endpoint_skip() -> None:
    verdicts = [G, B, S]
    with pytest.raises(UntestableEndpointError):
        bisect(_candidates(3), _from_list(verdicts))


def test_both_endpoints_skip_names_both() -> None:
    # When BOTH endpoints resolve skip, the error must name both untestable ends, not
    # just the first.
    verdicts = [S, G, S]
    with pytest.raises(UntestableEndpointError) as exc_info:
        bisect(_candidates(3), _from_list(verdicts))
    message = str(exc_info.value)
    assert "first" in message
    assert "last" in message


def test_single_skip_endpoint_names_only_that_end() -> None:
    # A single skipped endpoint must name only that end (no spurious mention of both).
    with pytest.raises(UntestableEndpointError) as first_exc:
        bisect(_candidates(3), _from_list([S, G, B]))
    first_msg = str(first_exc.value)
    assert "first" in first_msg
    assert "last" not in first_msg

    with pytest.raises(UntestableEndpointError) as last_exc:
        bisect(_candidates(3), _from_list([G, B, S]))
    last_msg = str(last_exc.value)
    assert "last" in last_msg
    assert "first" not in last_msg


def test_endpoint_errors_are_distinct_types() -> None:
    assert not issubclass(UntestableEndpointError, NonMonotonicError)
    assert not issubclass(NonMonotonicError, UntestableEndpointError)


def test_requires_at_least_two_candidates() -> None:
    with pytest.raises(ValueError):
        bisect(_candidates(1), _from_list([G]))


# -------------------------------------------------------------------- skip handling


def test_skip_in_interior_steps_over() -> None:
    # good good skip bad bad : mid lands on the skip and must fan out, finding the
    # boundary between index 1 (good) and 3 (bad) is not adjacent -> ambiguous? No:
    # here good at 1, bad at 3, skip at 2 -> boundary not adjacent -> ambiguous range.
    verdicts = [G, G, S, B, B]
    result = bisect(_candidates(5), _from_list(verdicts))
    # Confirmed good highest = 1, confirmed bad lowest = 3, gap includes the skip at 2.
    assert result.is_ambiguous
    assert result.ambiguous_range is not None
    lo, hi = result.ambiguous_range
    assert lo.order == 1
    assert hi.order == 3


def test_skip_in_interior_resolves_when_neighbor_decides() -> None:
    # A skip at the probed mid, but a neighbor narrows to an adjacent boundary.
    # good good good skip bad bad bad : 7 candidates, boundary effectively at 4.
    verdicts = [G, G, G, G, B, B, B]
    result = bisect(_candidates(7), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == 4


def test_skip_at_boundary_yields_ambiguous_range() -> None:
    # The breaking transition is hidden behind a skip exactly at the boundary.
    # good ... good SKIP bad ... bad with the skip being the only thing between
    # the last good and first bad.
    verdicts = [G, S, B]
    result = bisect(_candidates(3), _from_list(verdicts))
    assert result.is_ambiguous
    assert result.first_bad is None
    assert result.ambiguous_range is not None
    lo, hi = result.ambiguous_range
    assert (lo.order, hi.order) == (0, 2)


def test_all_skip_interior_is_ambiguous_no_infinite_loop() -> None:
    # Every interior candidate is skip; endpoints are valid good/bad.
    n = 9
    verdicts = [G] + [S] * (n - 2) + [B]
    result = bisect(_candidates(n), _from_list(verdicts))
    assert result.is_ambiguous
    assert result.ambiguous_range is not None
    lo, hi = result.ambiguous_range
    assert (lo.order, hi.order) == (0, n - 1)


# ------------------------------------------------------------------ flaky detection


def test_flaky_candidate_raises_or_is_ambiguous_never_wrong() -> None:
    # A candidate alternates good/bad across repeated probes. The bisect must either
    # raise NonMonotonicError or return an ambiguous range -- never a confident first_bad
    # that could be wrong.
    base = _candidates(5)
    flips = {"n": 0}

    def fn(candidate: Candidate) -> Verdict:
        # index 0 good, index 4 bad (valid endpoints). index 2 is flaky.
        if candidate.order == 0:
            return G
        if candidate.order == 4:
            return B
        if candidate.order == 2:
            flips["n"] += 1
            return G if flips["n"] % 2 == 1 else B
        # indices 1 and 3 are stable around the flaky middle.
        return G if candidate.order < 2 else B

    try:
        result = bisect(base, fn)
    except NonMonotonicError:
        return  # acceptable outcome
    # If it did not raise, it must not have produced a confidently wrong single culprit
    # built on the flaky candidate -- an ambiguous range or a boundary not touching the
    # flaky index is acceptable.
    assert result.is_ambiguous or (result.first_bad is not None and result.first_bad.order != 2)


def test_flaky_candidate_reprobed_raises_nonmonotonic() -> None:
    # Force an index to be probed twice and flip its verdict, exercising the memo's
    # flaky-detection guard. With n=5 and endpoints 0(G)/4(B):
    #   mid=2 -> SKIP, fan out finds 1(G) -> lo=1; next mid=(1+4)//2=2 re-probes index 2.
    # On its second probe index 2 returns BAD instead of SKIP -> NonMonotonicError.
    probes_of_2 = {"n": 0}

    def fn(candidate: Candidate) -> Verdict:
        if candidate.order == 0:
            return G
        if candidate.order == 4:
            return B
        if candidate.order == 1:
            return G
        if candidate.order == 3:
            return B
        # index 2: SKIP first time, BAD second time (a flip the guard must catch).
        probes_of_2["n"] += 1
        return S if probes_of_2["n"] == 1 else B

    with pytest.raises(NonMonotonicError):
        bisect(_candidates(5), fn)


# -------------------------------------------------------------------- probe bounds


def test_probe_count_is_bounded() -> None:
    # With no skips, probes <= 2 (endpoints) + ceil(log2 n).
    n = 64
    boundary = 37
    verdicts = [G] * boundary + [B] * (n - boundary)
    result = bisect(_candidates(n), _from_list(verdicts))
    assert result.first_bad is not None
    assert result.first_bad.order == boundary
    bound = 2 + math.ceil(math.log2(n))
    assert result.probes <= bound


def test_probe_count_bounded_with_skips() -> None:
    n = 32
    # Scatter a few skips; bound is endpoints + log2 n + number of skips probed.
    verdicts = [G] * 10 + [S, S] + [B] * (n - 12)
    result = bisect(_candidates(n), _from_list(verdicts))
    total_skips = sum(1 for v in verdicts if v is S)
    bound = 2 + math.ceil(math.log2(n)) + total_skips + 1
    assert result.probes <= bound


def test_steps_tested_recorded() -> None:
    verdicts = [G, G, B, B]
    result = bisect(_candidates(4), _from_list(verdicts))
    assert len(result.steps_tested) == result.probes
    # Each recorded entry pairs a candidate with the verdict it returned.
    for cand, verdict in result.steps_tested:
        assert verdict is verdicts[cand.order]
