"""Agent runner adapter.

agentbisect is not an agent framework; it wraps a user's agent behind the
:class:`AgentRunner` protocol. A runner is given a config and a tool provider and
must produce a :class:`~agentbisect.types.Trace`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import AgentConfig, Trace
from .tools import ToolProvider

__all__ = ["AgentRunner", "ToolProvider"]


@runtime_checkable
class AgentRunner(Protocol):
    """Adapter that drives a user's agent for one run.

    Implementations must call tools *only* through the supplied ``tools`` provider
    (so capture can record and replay can mock them) and must assign each emitted
    step a stable, contiguous 0-based ``index``.
    """

    def run(self, config: AgentConfig, tools: ToolProvider) -> Trace:
        """Execute one agent run under ``config`` and return its trace."""
        ...
