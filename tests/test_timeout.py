"""Tests for the per-candidate timeout helper. No real sleeps."""

from __future__ import annotations

import pytest

from agentbisect.timeout import (
    CandidateTimeout,
    FakeClock,
    expire_after,
    run_with_timeout,
    thread_timeout,
)


def test_none_and_zero_are_no_limit() -> None:
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        return 7

    assert run_with_timeout(fn, None) == 7
    assert run_with_timeout(fn, 0) == 7
    assert run_with_timeout(fn, 0.0) == 7
    assert calls["n"] == 3


def test_expire_after_overrun_raises_without_calling_fn() -> None:
    def fn() -> str:
        raise AssertionError("must not run: timeout should fail instantly")

    with pytest.raises(CandidateTimeout, match=r"0\.5"):
        run_with_timeout(fn, 0.5, impl=expire_after(2.0))


def test_expire_after_under_deadline_calls_fn() -> None:
    assert run_with_timeout(lambda: "ok", 5.0, impl=expire_after(0.1)) == "ok"


def test_fake_clock_jumps_without_sleeping() -> None:
    clock = FakeClock()
    impl = expire_after(1.25, clock=clock)
    with pytest.raises(CandidateTimeout):
        impl(lambda: None, 1.0)
    assert clock.now == 1.25


def test_thread_timeout_fast_fn_returns() -> None:
    assert thread_timeout(lambda: 42, 5.0) == 42


def test_thread_timeout_propagates_fn_error() -> None:
    def boom() -> int:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        thread_timeout(boom, 5.0)
