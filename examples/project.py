"""Example project config for the ``agentbisect`` CLI (fully offline).

Defines the hooks the CLI calls: ``runner``, ``config``, ``tool_executor``, ``oracle``.
Used by, e.g.::

    agentbisect capture --config examples/project.py --out /tmp/bundle --label demo
"""

from __future__ import annotations

from typing import Any

from agentbisect.agent.fake import FakeAgent
from agentbisect.oracle import AssertionOracle
from agentbisect.types import AgentConfig, RunBundle, Trace


def runner() -> FakeAgent:
    """The agent adapter (a deterministic, offline fake for this demo)."""
    return FakeAgent()


def config() -> AgentConfig:
    """The agent configuration to capture."""
    return AgentConfig(
        system_prompt="You are support. Always state the refund policy clearly.",
        model="support-bot",
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund={prompt_has:refund}",
        },
    )


def tool_executor() -> Any:
    """Performs real tool calls during capture (deterministic here)."""

    def run(tool: str, args: dict[str, Any]) -> Any:
        return f"{tool}-doc"

    return run


def oracle() -> AssertionOracle:
    """GOOD iff the agent's answer still reports the refund clause present."""

    def predicate(trace: Trace, _bundle: RunBundle) -> bool:
        return trace.final_output == "refund=yes"

    return AssertionOracle(predicate)
