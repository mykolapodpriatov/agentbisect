"""Deterministically replay a candidate config against a recorded trace.

The only thing allowed to vary between capture and replay is the axis under test:

* tool calls are served from the recording by :class:`MockToolProvider` (tools never
  re-execute), and
* every LLM step is forced to ``temperature=0`` (and ``seed`` where supported), because
  the model/prompt is often the axis under test and a flaky baseline cannot be cleanly
  bisected. A non-zero captured temperature triggers a :class:`ReplayTemperatureWarning`.

The function returns a :class:`~agentbisect.types.ReplayResult` (not a bare trace) so the
bisect engine can apply the quarantine rule to ``diverged``/``nearest`` results.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from .agent import AgentRunner
from .mock_tools import DivergenceError, DivergencePolicy, MockToolProvider
from .types import AgentConfig, ReplayResult, Trace

__all__ = ["ReplayTemperatureWarning", "forced_determinism_params", "replay"]


class ReplayTemperatureWarning(UserWarning):
    """Emitted when the captured config used a non-zero temperature.

    A flaky (``temperature > 0``) baseline cannot be cleanly bisected; replay forces
    ``temperature=0`` and surfaces this so the non-determinism is never silent.
    """


def forced_determinism_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return ``params`` with determinism forced for replay.

    Sets ``temperature=0`` and, when a ``seed`` is present in the source params, keeps
    it (providers that support it gain extra determinism; others ignore it).
    """
    forced = dict(params)
    forced["temperature"] = 0
    return forced


def replay(
    runner: AgentRunner,
    candidate_config: AgentConfig,
    recorded_trace: Trace,
    *,
    policy: DivergencePolicy = DivergencePolicy.SKIP,
    passthrough_executor: Callable[[str, dict[str, Any]], Any] | None = None,
) -> ReplayResult:
    """Replay ``candidate_config`` with tools mocked from ``recorded_trace``.

    Parameters
    ----------
    runner:
        The user's agent adapter.
    candidate_config:
        The config to evaluate (one axis overridden vs. the captured config).
    recorded_trace:
        The trace whose tool outputs are replayed.
    policy:
        Divergence policy for tool calls with no in-order recorded match.
    passthrough_executor:
        Required only when ``policy`` is ``PASSTHROUGH``; performs live tool calls.

    Returns
    -------
    ReplayResult
        The replayed trace plus divergence/substitution/passthrough flags.
    """
    captured_temp = candidate_config.params.get("temperature", 0)
    if isinstance(captured_temp, (int, float)) and captured_temp != 0:
        warnings.warn(
            f"captured config used temperature={captured_temp!r}; forcing temperature=0 for "
            "a reproducible replay (a flaky baseline cannot be cleanly bisected)",
            ReplayTemperatureWarning,
            stacklevel=2,
        )

    forced_config = candidate_config.with_overrides(
        params=forced_determinism_params(candidate_config.params)
    )
    provider = MockToolProvider(
        recorded_trace,
        policy=policy,
        passthrough_executor=passthrough_executor,
    )

    try:
        trace = runner.run(forced_config, provider)
    except DivergenceError as exc:
        # SKIP / unmatched divergence: abort with a flagged, empty-ish result.
        partial = Trace(final_output="")
        return ReplayResult(
            trace=partial,
            diverged=True,
            notes=(str(exc),),
        )

    return ReplayResult(
        trace=trace,
        diverged=provider.diverged,
        has_nearest_substitutions=provider.has_nearest_substitutions,
        used_passthrough=provider.used_passthrough,
        notes=tuple(provider.notes),
    )
