"""Tool providers for capture and replay.

* :class:`RecordingToolProvider` runs real tools and records each call so capture can
  build a :class:`~agentbisect.types.Trace` with recorded I/O.
* :class:`MockToolProvider` serves recorded outputs during replay, keyed by
  ``(tool, canonical args, occurrence)`` in call order. When a candidate makes a call
  with no in-order recorded match (the agent diverged), a documented
  :class:`DivergencePolicy` decides what happens.

The occurrence key is what makes repeated/retried identical calls map unambiguously to
their k-th recorded output instead of all sharing one output.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from .canonical import canonical_json
from .types import ToolStep, Trace

__all__ = [
    "DivergenceError",
    "DivergencePolicy",
    "MockToolProvider",
    "RecordingToolProvider",
]


class DivergencePolicy(StrEnum):
    """What to do when a replayed tool call has no in-order recorded match.

    * ``SKIP`` (default): mark the replay ``diverged`` and abort the call -- the
      candidate is untestable and the bisect quarantines it as ``skip``.
    * ``NEAREST``: serve the recorded output for the same tool whose recorded args are
      *closest* to the unmatched call's args under :func:`_arg_distance` (off by
      default); sets ``has_nearest_substitutions`` so the bisect still quarantines it.
    * ``PASSTHROUGH``: re-execute the real tool (explicit opt-in); sets
      ``used_passthrough`` -- NOT quarantined, but surfaced so live I/O is never silent.
    """

    SKIP = "skip"
    NEAREST = "nearest"
    PASSTHROUGH = "passthrough"


class DivergenceError(RuntimeError):
    """Raised internally to abort a replay when the SKIP policy hits a divergence.

    Caught by :func:`agentbisect.replay.replay`, which converts it into a
    ``ReplayResult(diverged=True)``.
    """

    def __init__(self, tool: str, args: dict[str, Any]) -> None:
        self.tool = tool
        self.call_args = args
        super().__init__(f"diverged: no recorded output for tool {tool!r} with args {args!r}")


def _key(tool: str, args: dict[str, Any]) -> tuple[str, str]:
    """Canonical match key for a tool call (occurrence is tracked separately)."""
    return (tool, canonical_json(args))


def _arg_distance(query: dict[str, Any], recorded: dict[str, Any]) -> int:
    """Return the number of differing argument fields between two arg dicts.

    The distance is the count, over the *union* of keys, of fields that differ: a key
    present in only one dict counts as one difference, and a key present in both whose
    canonicalized values differ counts as one difference. Identical canonical args give a
    distance of ``0``. This is a deliberately simple, explainable metric -- the NEAREST
    output is only an escape hatch (the bisect quarantines it as ``skip``), so it just has
    to be *honest* about which recorded call is closest, not a semantic similarity model.
    """

    def _differs(field: str) -> bool:
        if field not in query or field not in recorded:
            return True
        return canonical_json(query[field]) != canonical_json(recorded[field])

    return sum(_differs(field) for field in query.keys() | recorded.keys())


class RecordingToolProvider:
    """Executes real tools via ``executor`` and records every call as a ``ToolStep``."""

    def __init__(self, executor: Callable[[str, dict[str, Any]], Any]) -> None:
        self._executor = executor
        self._seen: dict[tuple[str, str], int] = {}
        self.records: list[ToolStep] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        """Run the real tool, record the call with its occurrence, return the output."""
        output = self._executor(tool, args)
        key = _key(tool, args)
        occurrence = self._seen.get(key, 0)
        self._seen[key] = occurrence + 1
        # index is patched by capture() against the full step order; provisional here.
        self.records.append(
            ToolStep(
                index=len(self.records),
                tool=tool,
                args=dict(args),
                output=output,
                ok=True,
                occurrence=occurrence,
            )
        )
        return output


class MockToolProvider:
    """Serves recorded tool outputs during replay, in call order, by occurrence.

    Attributes are read by :func:`agentbisect.replay.replay` after the run to populate
    the :class:`~agentbisect.types.ReplayResult` flags.
    """

    def __init__(
        self,
        recorded: Trace,
        policy: DivergencePolicy = DivergencePolicy.SKIP,
        passthrough_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._policy = policy
        self._passthrough_executor = passthrough_executor
        # Bucketed recorded outputs: (tool, canon_args) -> [outputs in occurrence order].
        self._buckets: dict[tuple[str, str], list[Any]] = {}
        # Recorded (args, output) pairs per tool (for NEAREST), in recorded order so that
        # ties in arg-distance resolve to the earliest recorded call.
        self._by_tool: dict[str, list[tuple[dict[str, Any], Any]]] = {}
        for step in recorded.tool_steps():
            self._buckets.setdefault(_key(step.tool, step.args), []).append(step.output)
            self._by_tool.setdefault(step.tool, []).append((dict(step.args), step.output))
        # Per-key consumption cursor (which occurrence to serve next).
        self._cursor: dict[tuple[str, str], int] = {}
        self.diverged: bool = False
        self.has_nearest_substitutions: bool = False
        self.used_passthrough: bool = False
        self.notes: list[str] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        """Return the next recorded output for this exact call, or apply the policy."""
        key = _key(tool, args)
        bucket = self._buckets.get(key, [])
        occurrence = self._cursor.get(key, 0)
        if occurrence < len(bucket):
            self._cursor[key] = occurrence + 1
            return bucket[occurrence]
        return self._handle_divergence(tool, args)

    def _handle_divergence(self, tool: str, args: dict[str, Any]) -> Any:
        """Apply the configured divergence policy for an unmatched call."""
        if self._policy is DivergencePolicy.PASSTHROUGH:
            if self._passthrough_executor is None:
                raise DivergenceError(tool, args)
            self.used_passthrough = True
            self.notes.append(f"passthrough: re-executed live tool {tool!r}")
            return self._passthrough_executor(tool, args)
        if self._policy is DivergencePolicy.NEAREST:
            candidates = self._by_tool.get(tool)
            if candidates:
                self.has_nearest_substitutions = True
                # Serve the recorded call whose args are closest; ties -> earliest, which
                # ``min`` preserves because the list is in recorded order and the key is
                # the distance alone (a stable sort over a single comparable).
                rec_args, output = min(candidates, key=lambda pair: _arg_distance(args, pair[0]))
                self.notes.append(
                    f"nearest: served the closest recorded {tool!r} output "
                    f"(args distance {_arg_distance(args, rec_args)})"
                )
                return output
            # No recorded output for this tool at all -> genuine divergence.
        # SKIP (default) or NEAREST with nothing to substitute.
        self.diverged = True
        self.notes.append(f"diverged: no in-order recorded output for {tool!r}")
        raise DivergenceError(tool, args)
