"""Capture a real agent run into a recorded trace.

``capture`` drives the user's :class:`~agentbisect.agent.AgentRunner` for real,
executing tools via a :class:`~agentbisect.mock_tools.RecordingToolProvider` so every
tool call's I/O is recorded keyed by ``(tool, canonical args, occurrence)``. The result
is wrapped in a :class:`~agentbisect.types.RunBundle` for portable, versioned storage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent import AgentRunner
from .bundle import make_bundle
from .mock_tools import RecordingToolProvider
from .types import AgentConfig, RunBundle, Trace

__all__ = ["capture"]


def capture(
    runner: AgentRunner,
    config: AgentConfig,
    tool_executor: Callable[[str, dict[str, Any]], Any],
    *,
    label: str,
) -> RunBundle:
    """Run the agent for real and return a versioned bundle of the run.

    Parameters
    ----------
    runner:
        The user's agent adapter.
    config:
        The agent configuration to run.
    tool_executor:
        Callable that performs real tool calls (``(tool, args) -> output``); its
        outputs are recorded for deterministic replay.
    label:
        Caller-supplied label for the bundle. The library never reads the clock, so
        the caller owns any timestamp/identifier here.

    Returns
    -------
    RunBundle
        The captured config + trace with an integrity hash.
    """
    provider = RecordingToolProvider(tool_executor)
    trace = runner.run(config, provider)
    # The runner is responsible for contiguous indices; we trust the trace's order but
    # rebuild the final trace from the runner's own steps (which already carry indices).
    captured = Trace(steps=trace.steps, final_output=trace.final_output)
    return make_bundle(config=config, trace=captured, label=label)
