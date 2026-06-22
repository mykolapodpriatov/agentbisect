"""Core data types for agentbisect.

Everything here is a frozen pydantic model (immutable, validated) so that captured
bundles, candidates, and bisect results are safe to share, hash, and serialize.

Key invariants:

* :class:`Step` carries a stable 0-based ``index`` giving the total order of the run.
  :class:`ToolStep` additionally carries an ``occurrence`` (the k-th call of the same
  ``(tool, canonical_args)``) so that repeated/retried calls map unambiguously to their
  recorded outputs during replay.
* :class:`AgentConfig` exposes a deterministic :meth:`AgentConfig.fingerprint`.
* :class:`RunBundle` stores an ``integrity`` hash of ``config`` + ``trace`` so tampering
  is detectable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .canonical import stable_hash

__all__ = [
    "BUNDLE_VERSION",
    "AgentConfig",
    "BisectResult",
    "Candidate",
    "LlmStep",
    "ReplayResult",
    "RunBundle",
    "Step",
    "ToolSchema",
    "ToolStep",
    "Trace",
    "Verdict",
]

#: Current on-disk bundle schema version. Bumped when the serialized shape changes.
BUNDLE_VERSION = 1


class _Frozen(BaseModel):
    """Base for immutable, strictly-validated models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSchema(_Frozen):
    """Declaration of a tool the agent may call.

    The JSON schema is stored on ``json_schema`` but accepted and serialized under the
    name ``schema`` (its natural name; a plain ``schema`` field would shadow pydantic's
    ``BaseModel.schema``). Use ``ToolSchema(name=..., schema={...})`` as usual.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str
    json_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    version: str = "1"

    @property
    def schema(self) -> dict[str, Any]:  # type: ignore[override]
        """The tool's JSON schema (alias for ``json_schema``)."""
        return self.json_schema


class AgentConfig(_Frozen):
    """The full, versioned configuration of an agent run.

    An agent run is treated as a function of this config plus the recorded tool
    outputs. A bisect overrides exactly one axis of this config per candidate.
    """

    system_prompt: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    tool_schemas: tuple[ToolSchema, ...] = ()
    retrieval_ref: str | None = None

    def fingerprint(self) -> str:
        """Return a stable hash of this config under the determinism discipline."""
        return stable_hash(
            {
                "system_prompt": self.system_prompt,
                "model": self.model,
                "params": self.params,
                "tool_schemas": [
                    {"name": ts.name, "schema": ts.json_schema, "version": ts.version}
                    for ts in self.tool_schemas
                ],
                "retrieval_ref": self.retrieval_ref,
            }
        )

    def with_overrides(self, **changes: Any) -> AgentConfig:
        """Return a copy of this config with ``changes`` applied (single-axis isolation)."""
        return self.model_copy(update=changes)


class LlmStep(_Frozen):
    """A single LLM turn in the trace."""

    kind: Literal["llm"] = "llm"
    index: int
    role: str
    content: str
    tool_calls: tuple[str, ...] = ()


class ToolStep(_Frozen):
    """A single tool invocation in the trace.

    ``occurrence`` is the 0-based count of how many *prior* steps invoked the same
    ``(tool, canonical args)`` pair, so the k-th identical call maps to the k-th
    recorded output.
    """

    kind: Literal["tool"] = "tool"
    index: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    ok: bool = True
    occurrence: int = 0


#: A step is either an LLM turn or a tool call, discriminated on ``kind``.
Step = LlmStep | ToolStep


class Trace(_Frozen):
    """An ordered sequence of steps plus the agent's final output."""

    steps: tuple[Step, ...] = ()
    final_output: str = ""

    def digest(self) -> str:
        """Return a stable hash of the whole trace (used by the judge cache key)."""
        return stable_hash(
            {
                "steps": [s.model_dump(mode="json") for s in self.steps],
                "final_output": self.final_output,
            }
        )

    def tool_steps(self) -> tuple[ToolStep, ...]:
        """Return only the tool steps, in index order."""
        return tuple(s for s in self.steps if isinstance(s, ToolStep))


class ReplayResult(_Frozen):
    """The outcome of replaying a candidate config against a recorded trace.

    The bisect engine consumes this (not a bare trace) so it can apply the quarantine
    rule: a ``diverged`` or ``has_nearest_substitutions`` result is treated as ``skip``.
    """

    trace: Trace
    #: A tool call had no in-order recorded match (the agent genuinely diverged).
    diverged: bool = False
    #: The ``nearest`` policy served a non-exact recorded output for some call.
    has_nearest_substitutions: bool = False
    #: A real tool was re-executed (explicit ``passthrough`` opt-in).
    used_passthrough: bool = False
    #: Human-readable notes (e.g. the divergence point) for the report.
    notes: tuple[str, ...] = ()


class RunBundle(_Frozen):
    """A portable, versioned capture of an agent run.

    Stores the config, the recorded trace, a caller-supplied ``label`` (the library
    never reads the clock), and an ``integrity`` hash of ``config`` + ``trace`` so
    tampering with a stored bundle is detectable on load.
    """

    version: int
    config: AgentConfig
    trace: Trace
    label: str = ""
    integrity: str = ""


class Verdict(StrEnum):
    """The oracle's judgement of a replayed candidate."""

    GOOD = "good"
    BAD = "bad"
    SKIP = "skip"


class Candidate(_Frozen):
    """One point on an axis: the captured config with a single axis overridden."""

    axis: str
    ref: str
    config: AgentConfig
    order: int


class BisectResult(_Frozen):
    """The result of a bisection run.

    ``first_bad`` is populated *only* for a directly-adjacent confirmed
    good->bad transition. When the boundary is undetermined (skips at the edge, an
    all-skip interval) ``first_bad`` is ``None`` and ``ambiguous_range`` holds the
    bracketing ``[last_good, first_bad_or_unknown]`` candidates instead.
    """

    axis: str
    first_bad: Candidate | None
    last_good: Candidate | None
    ambiguous_range: tuple[Candidate, Candidate] | None = None
    steps_tested: tuple[tuple[Candidate, Verdict], ...] = ()
    probes: int = 0

    @property
    def is_ambiguous(self) -> bool:
        """True when no single first-bad change could be isolated."""
        return self.first_bad is None
