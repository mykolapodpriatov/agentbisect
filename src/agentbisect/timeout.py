"""Per-candidate timeout for replay / verdict calls.

``0`` or ``None`` means no limit (today's behaviour). Production waits via a daemon
thread + ``join``; tests inject :func:`expire_after` (a fake clock that jumps) so the
suite never sleeps. An overrun raises :class:`CandidateTimeout` -- callers quarantine
that as ``skip``, never as ``bad``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = [
    "CandidateTimeout",
    "FakeClock",
    "TimeoutImpl",
    "expire_after",
    "run_with_timeout",
    "thread_timeout",
]

T = TypeVar("T")

#: A wait strategy: run ``fn`` with a positive deadline, or raise CandidateTimeout.
TimeoutImpl = Callable[[Callable[[], Any], float], Any]


class CandidateTimeout(Exception):
    """A per-candidate replay/verdict exceeded its ``--timeout``."""


class FakeClock:
    """Monotonic clock stub. Tests :meth:`advance` it instead of sleeping."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Jump forward ``seconds`` (may be zero). Never sleeps."""
        self.now += seconds


def expire_after(elapsed: float, clock: FakeClock | None = None) -> TimeoutImpl:
    """Return a timeout impl that pretends ``elapsed`` seconds of work, then checks.

    If ``elapsed >= seconds`` it raises :class:`CandidateTimeout` without calling
    ``fn``. Otherwise it calls ``fn``. Never sleeps.
    """
    clk = clock if clock is not None else FakeClock()

    def impl(fn: Callable[[], Any], seconds: float) -> Any:
        start = clk()
        clk.advance(elapsed)
        if clk() - start >= seconds:
            raise CandidateTimeout(f"candidate timed out after {seconds}s")
        return fn()

    return impl


def thread_timeout(fn: Callable[[], T], seconds: float) -> T:
    """Production wait: daemon thread + ``join(seconds)``. Does not kill the worker."""
    box: list[T] = []
    errors: list[Exception] = []

    def target() -> None:
        try:
            box.append(fn())
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise CandidateTimeout(f"candidate timed out after {seconds}s")
    if errors:
        raise errors[0]
    if not box:
        raise CandidateTimeout(f"candidate timed out after {seconds}s")
    return box[0]


def run_with_timeout(
    fn: Callable[[], T],
    seconds: float | None,
    *,
    impl: TimeoutImpl | None = None,
) -> T:
    """Run ``fn`` with an optional deadline of ``seconds``.

    ``None`` or ``<= 0`` means no limit and ``fn`` is called directly. A positive
    deadline uses ``impl`` (default :func:`thread_timeout`).
    """
    if seconds is None or seconds <= 0:
        return fn()
    if impl is None:
        return thread_timeout(fn, seconds)
    return cast(T, impl(fn, seconds))
