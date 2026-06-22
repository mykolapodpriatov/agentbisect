"""Tool provider protocol used by runners during capture and replay.

During *capture* the provider executes real tools and records their I/O. During
*replay* a :class:`~agentbisect.mock_tools.MockToolProvider` serves recorded outputs
keyed by ``(tool, canonical args, occurrence)`` so tools never re-execute.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["ToolProvider"]


@runtime_checkable
class ToolProvider(Protocol):
    """Serves tool calls to a running agent.

    A runner calls :meth:`call` for every tool invocation. The provider returns the
    tool output and, under the hood, records (capture) or matches (replay) the call.
    """

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        """Invoke ``tool`` with ``args`` and return its output."""
        ...
