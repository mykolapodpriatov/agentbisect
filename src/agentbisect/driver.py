"""Wire replay -> oracle -> quarantine into a verdict function, and run a full bisection.

The pure :func:`agentbisect.bisect.bisect` core takes a ``verdict_fn`` that maps a
candidate to good/bad/skip. :func:`make_verdict_fn` builds that function from a runner +
recorded trace + oracle and applies the **quarantine rule**: a :class:`ReplayResult`
that ``diverged`` or used a ``nearest`` substitution is forced to ``skip`` regardless of
the oracle's opinion, so a verdict is never built on a fabricated tool output.
``passthrough`` results are *not* quarantined (the user opted in) but are surfaced.

:func:`run_bisection` ties it together end-to-end and additionally computes the minimal
repro and behavioral diff for the report when a single first-bad change is found. The
post-bisect artifact replays are themselves subject to the quarantine rule: if a second
replay of ``first_bad`` (or ``last_good``) diverges or substitutes a ``nearest`` output,
the artifacts are reported as *unavailable* rather than built from a fabricated/empty
trace, so a flaky replay can never corrupt ``minimal_repro`` or ``behavioral_diff``.
``first_bad`` itself is unaffected (it comes from the completed search).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from .agent import AgentRunner
from .bisect import bisect
from .diff import BehavioralDiff, diff
from .mock_tools import DivergencePolicy
from .oracle import Oracle
from .replay import replay
from .types import BisectResult, Candidate, ReplayResult, RunBundle, Trace, Verdict

__all__ = [
    "BisectionOutcome",
    "ReplayDivergedWarning",
    "make_verdict_fn",
    "run_bisection",
]


class ReplayDivergedWarning(UserWarning):
    """Emitted when a post-bisect artifact replay diverged or used a nearest output.

    The behavioral diff and minimal repro are computed from a *second* replay of the
    isolated endpoints. If that replay is non-reproducible, the artifacts are marked
    unavailable rather than built from a corrupt/empty trace; this warning surfaces the
    non-determinism so it is never silent. The ``first_bad`` result still stands.
    """


def make_verdict_fn(
    runner: AgentRunner,
    bundle: RunBundle,
    oracle: Oracle,
    *,
    policy: DivergencePolicy = DivergencePolicy.SKIP,
    passthrough_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    on_passthrough: Callable[[Candidate, ReplayResult], None] | None = None,
) -> Callable[[Candidate], Verdict]:
    """Return a verdict function for the bisect core, folding in the quarantine rule.

    A candidate is replayed against the bundle's recorded trace; the oracle judges the
    replayed trace. If the replay ``diverged`` or substituted a ``nearest`` output, the
    verdict is forced to ``skip``. ``passthrough`` replays are surfaced via
    ``on_passthrough`` but their oracle verdict stands.
    """

    def verdict_fn(candidate: Candidate) -> Verdict:
        result = replay(
            runner,
            candidate.config,
            bundle.trace,
            policy=policy,
            passthrough_executor=passthrough_executor,
        )
        if result.diverged or result.has_nearest_substitutions:
            return Verdict.SKIP
        if result.used_passthrough and on_passthrough is not None:
            on_passthrough(candidate, result)
        return oracle.judge(result.trace, bundle)

    return verdict_fn


class BisectionOutcome:
    """The full result of a bisection plus repro/diff artifacts for reporting.

    ``artifacts_unavailable`` is set (to a human-readable reason) when the post-bisect
    artifact replay diverged, leaving ``minimal_repro`` and ``behavioral_diff`` as
    ``None``; the ``result`` (and its ``first_bad``) is still valid.
    """

    def __init__(
        self,
        result: BisectResult,
        *,
        minimal_repro: Trace | None,
        behavioral_diff: BehavioralDiff | None,
        used_passthrough: bool,
        artifacts_unavailable: str | None = None,
    ) -> None:
        self.result = result
        self.minimal_repro = minimal_repro
        self.behavioral_diff = behavioral_diff
        self.used_passthrough = used_passthrough
        self.artifacts_unavailable = artifacts_unavailable


def _replay_for_artifacts(
    runner: AgentRunner,
    candidate: Candidate,
    bundle: RunBundle,
    policy: DivergencePolicy,
) -> ReplayResult:
    """Replay a candidate for artifact building, returning the full quarantine-aware result.

    The caller must honor the quarantine rule (a ``diverged``/``has_nearest_substitutions``
    result is non-reproducible) instead of forwarding a corrupt/empty trace to ``diff`` or
    ``minimize``.
    """
    return replay(runner, candidate.config, bundle.trace, policy=policy)


def run_bisection(
    runner: AgentRunner,
    bundle: RunBundle,
    candidates: list[Candidate],
    oracle: Oracle,
    *,
    policy: DivergencePolicy = DivergencePolicy.SKIP,
) -> BisectionOutcome:
    """Run a full bisection over ``candidates`` and assemble report artifacts.

    When a single first-bad change is isolated, computes the behavioral diff between the
    last-good and first-bad replays and a minimal reproducing trace of the first-bad run.
    """
    passthrough_seen = {"value": False}

    def _note_passthrough(_c: Candidate, _r: ReplayResult) -> None:
        passthrough_seen["value"] = True

    verdict_fn = make_verdict_fn(
        runner,
        bundle,
        oracle,
        policy=policy,
        on_passthrough=_note_passthrough,
    )
    result = bisect(candidates, verdict_fn)

    minimal_repro: Trace | None = None
    behavioral_diff: BehavioralDiff | None = None
    artifacts_unavailable: str | None = None

    if result.first_bad is not None and result.last_good is not None:
        good = _replay_for_artifacts(runner, result.last_good, bundle, policy)
        bad = _replay_for_artifacts(runner, result.first_bad, bundle, policy)

        # Apply the quarantine rule to the artifact replays too: a diverged or
        # nearest-substituted replay is non-reproducible, so never feed its trace to
        # diff()/minimize() (that would corrupt the artifacts). Mark them unavailable.
        unreproducible = (
            good.diverged
            or good.has_nearest_substitutions
            or bad.diverged
            or bad.has_nearest_substitutions
        )
        if unreproducible:
            artifacts_unavailable = (
                "unavailable (replay diverged): a post-bisect replay of the isolated "
                "endpoints did not reproduce deterministically, so the behavioral diff "
                "and minimal repro were not built; the first-bad result still stands"
            )
            warnings.warn(artifacts_unavailable, ReplayDivergedWarning, stacklevel=2)
        else:
            behavioral_diff = diff(good.trace, bad.trace)

            from .minimize import minimize

            def trace_verdict(t: Trace) -> Verdict:
                return oracle.judge(t, bundle)

            minimal_repro = minimize(bad.trace, trace_verdict)

    return BisectionOutcome(
        result,
        minimal_repro=minimal_repro,
        behavioral_diff=behavioral_diff,
        used_passthrough=passthrough_seen["value"],
        artifacts_unavailable=artifacts_unavailable,
    )
