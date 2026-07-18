"""Axis providers: each yields an ORDERED list of candidates (old -> new).

A bisect runs over exactly one axis at a time. Every candidate overrides exactly one
field of the captured config and holds all other fields at the bundle's values, so a
found culprit is attributable to that single axis (single-axis isolation).

Providers:

* :class:`ModelListAxis` -- order is the supplied list order.
* :class:`PromptGitAxis` -- order is git history (old -> new) of a prompt file; each
  candidate overrides ``system_prompt`` with that revision's file contents.
* :class:`ToolSchemaAxis` -- ordered tool-schema sets; overrides ``tool_schemas``.
* :class:`RetrievalAxis` -- ordered retrieval snapshot refs; overrides ``retrieval_ref``.
* :class:`ParamsAxis` -- ordered values for one sampling/params key; overrides
  ``params[key]`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import AgentConfig, Candidate, ToolSchema

__all__ = [
    "Axis",
    "ModelListAxis",
    "ParamsAxis",
    "PromptGitAxis",
    "RetrievalAxis",
    "ToolSchemaAxis",
]

#: Params keys that :func:`agentbisect.replay.forced_determinism_params` overrides
#: unconditionally during replay, so a :class:`ParamsAxis` over them would be a silent
#: no-op (every candidate replays identically). Bisecting them is rejected.
_REPLAY_FORCED_PARAMS = frozenset({"temperature"})


@runtime_checkable
class Axis(Protocol):
    """Yields an ordered candidate list given the captured base config."""

    name: str

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        """Return candidates (old -> new), each overriding only this axis."""
        ...


class ModelListAxis:
    """Candidates are the supplied model ids, in list order (old -> new)."""

    name = "model"

    def __init__(self, models: Sequence[str]) -> None:
        if not models:
            raise ValueError("ModelListAxis requires at least one model")
        self._models = list(models)

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        return [
            Candidate(
                axis=self.name,
                ref=model,
                config=base.with_overrides(model=model),
                order=i,
            )
            for i, model in enumerate(self._models)
        ]


class PromptGitAxis:
    """Candidates are git revisions of a prompt file, ordered old -> new.

    Each candidate overrides ``system_prompt`` with the file's contents at that revision.
    """

    name = "prompt"

    def __init__(self, repo_path: str | Path, file_path: str, rev: str = "HEAD") -> None:
        self._repo_path = Path(repo_path)
        self._file_path = file_path
        self._rev = rev

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        import git

        repo = git.Repo(self._repo_path)
        # iter_commits yields newest -> oldest; reverse for old -> new ordering.
        commits = list(repo.iter_commits(self._rev, paths=self._file_path))
        commits.reverse()
        if not commits:
            raise ValueError(
                f"no commits touch {self._file_path!r} in {self._repo_path} at rev {self._rev!r}"
            )

        result: list[Candidate] = []
        for order, commit in enumerate(commits):
            prompt = self._read_blob(commit, self._file_path)
            if prompt is None:
                # File did not exist at this commit (e.g. a deletion); skip it.
                continue
            result.append(
                Candidate(
                    axis=self.name,
                    ref=commit.hexsha,
                    config=base.with_overrides(system_prompt=prompt),
                    order=order,
                )
            )
        # Re-number order densely after any skips so indices stay contiguous.
        return [c.model_copy(update={"order": i}) for i, c in enumerate(result)]

    @staticmethod
    def _read_blob(commit: Any, path: str) -> str | None:
        """Return the file contents at ``commit`` or ``None`` if absent there."""
        try:
            blob = commit.tree / path
        except KeyError:
            return None
        data = blob.data_stream.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)


class ToolSchemaAxis:
    """Candidates are ordered tool-schema sets (old -> new), overriding ``tool_schemas``."""

    name = "tools"

    def __init__(self, versions: Sequence[Sequence[ToolSchema]]) -> None:
        if not versions:
            raise ValueError("ToolSchemaAxis requires at least one version")
        self._versions = [tuple(v) for v in versions]

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        result: list[Candidate] = []
        for i, schemas in enumerate(self._versions):
            ref = ",".join(f"{s.name}@{s.version}" for s in schemas) or "(none)"
            result.append(
                Candidate(
                    axis=self.name,
                    ref=ref,
                    config=base.with_overrides(tool_schemas=tuple(schemas)),
                    order=i,
                )
            )
        return result


class RetrievalAxis:
    """Candidates are ordered retrieval snapshot refs, overriding ``retrieval_ref``."""

    name = "retrieval"

    def __init__(self, snapshots: Sequence[str]) -> None:
        if not snapshots:
            raise ValueError("RetrievalAxis requires at least one snapshot")
        self._snapshots = list(snapshots)

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        return [
            Candidate(
                axis=self.name,
                ref=snapshot,
                config=base.with_overrides(retrieval_ref=snapshot),
                order=i,
            )
            for i, snapshot in enumerate(self._snapshots)
        ]


class ParamsAxis:
    """Candidates are ordered values for one sampling key, overriding ``params[key]``.

    Each candidate overrides *only* ``params[key]`` and holds every other params entry
    (and all other config fields) at the captured value, so a found culprit is
    attributable to that single key (single-axis isolation). ``ref`` is ``"{key}={value}"``.

    A key that :func:`agentbisect.replay.forced_determinism_params` overrides during
    replay (e.g. ``temperature``) is rejected: replay forces it to a fixed value for
    every candidate, so bisecting it would be a silent no-op.
    """

    name = "params"

    def __init__(self, key: str, values: Sequence[Any]) -> None:
        if key in _REPLAY_FORCED_PARAMS:
            raise ValueError(
                f"ParamsAxis cannot bisect {key!r}: replay forces it to a fixed value "
                "for a reproducible replay, so every candidate would replay identically "
                "(a silent no-op); choose a params key that is not forced for replay"
            )
        if not values:
            raise ValueError("ParamsAxis requires at least one value")
        self._key = key
        self._values = list(values)

    def candidates(self, base: AgentConfig) -> list[Candidate]:
        result: list[Candidate] = []
        for i, value in enumerate(self._values):
            params = {**base.params, self._key: value}
            result.append(
                Candidate(
                    axis=self.name,
                    ref=f"{self._key}={value}",
                    config=base.with_overrides(params=params),
                    order=i,
                )
            )
        return result
