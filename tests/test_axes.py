"""Tests for axis providers: ordering and single-axis isolation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agentbisect.axes import (
    ModelListAxis,
    ParamsAxis,
    PromptGitAxis,
    RetrievalAxis,
    ToolSchemaAxis,
)
from agentbisect.types import AgentConfig, ToolSchema


def _base() -> AgentConfig:
    return AgentConfig(
        system_prompt="ORIGINAL PROMPT",
        model="base-model",
        params={"temperature": 0},
        tool_schemas=(ToolSchema(name="t", version="1"),),
        retrieval_ref="base-snap",
    )


def test_model_list_axis_order_and_isolation() -> None:
    axis = ModelListAxis(["m0", "m1", "m2"])
    cands = axis.candidates(_base())
    assert [c.order for c in cands] == [0, 1, 2]
    assert [c.ref for c in cands] == ["m0", "m1", "m2"]
    # Only the model changed; everything else held at base values.
    for c in cands:
        assert c.config.system_prompt == "ORIGINAL PROMPT"
        assert c.config.retrieval_ref == "base-snap"
        assert c.axis == "model"
    assert [c.config.model for c in cands] == ["m0", "m1", "m2"]


def test_model_list_axis_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ModelListAxis([])


def test_retrieval_axis_order_and_isolation() -> None:
    axis = RetrievalAxis(["snap-a", "snap-b"])
    cands = axis.candidates(_base())
    assert [c.config.retrieval_ref for c in cands] == ["snap-a", "snap-b"]
    for c in cands:
        assert c.config.system_prompt == "ORIGINAL PROMPT"
        assert c.config.model == "base-model"


def test_tool_schema_axis_order_and_isolation() -> None:
    v0 = [ToolSchema(name="search", version="1")]
    v1 = [ToolSchema(name="search", version="2")]
    axis = ToolSchemaAxis([v0, v1])
    cands = axis.candidates(_base())
    assert cands[0].config.tool_schemas[0].version == "1"
    assert cands[1].config.tool_schemas[0].version == "2"
    assert cands[0].ref == "search@1"
    assert cands[1].ref == "search@2"
    for c in cands:
        assert c.config.system_prompt == "ORIGINAL PROMPT"


def test_prompt_git_axis_old_to_new(
    temp_git_repo: Callable[[str, list[str]], Path],
) -> None:
    versions = [
        "Always state the refund policy clearly.",
        "Always state the refund policy clearly. Be concise.",
        "Be concise.",  # the regression: dropped the refund clause
    ]
    repo = temp_git_repo("prompts/system.txt", versions)
    axis = PromptGitAxis(repo, "prompts/system.txt")
    cands = axis.candidates(_base())

    # Old -> new ordering, each overriding ONLY the system prompt.
    assert [c.order for c in cands] == [0, 1, 2]
    assert [c.config.system_prompt for c in cands] == versions
    for c in cands:
        assert c.config.model == "base-model"
        assert c.config.retrieval_ref == "base-snap"
        assert c.axis == "prompt"


def test_prompt_git_axis_no_commits_raises(
    temp_git_repo: Callable[[str, list[str]], Path],
) -> None:
    repo = temp_git_repo("prompts/system.txt", ["v0"])
    axis = PromptGitAxis(repo, "does/not/exist.txt")
    with pytest.raises(ValueError):
        axis.candidates(_base())


def test_prompt_git_axis_refs_are_commit_shas(
    temp_git_repo: Callable[[str, list[str]], Path],
) -> None:
    repo = temp_git_repo("p.txt", ["a", "b"])
    cands = PromptGitAxis(repo, "p.txt").candidates(_base())
    assert all(len(c.ref) == 40 for c in cands)  # full hex sha
    assert cands[0].ref != cands[1].ref


def test_tool_schema_axis_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ToolSchemaAxis([])


def test_tool_schema_axis_empty_version_ref() -> None:
    axis = ToolSchemaAxis([[]])  # a version with no schemas
    cands = axis.candidates(_base())
    assert cands[0].ref == "(none)"
    assert cands[0].config.tool_schemas == ()


def test_retrieval_axis_rejects_empty() -> None:
    with pytest.raises(ValueError):
        RetrievalAxis([])


def test_params_axis_order_and_isolation() -> None:
    axis = ParamsAxis("max_tokens", [256, 512, 1024])
    cands = axis.candidates(_base())
    # Candidates follow list order and expose the single key on each ref.
    assert [c.order for c in cands] == [0, 1, 2]
    assert [c.ref for c in cands] == [
        "max_tokens=256",
        "max_tokens=512",
        "max_tokens=1024",
    ]
    assert [c.config.params["max_tokens"] for c in cands] == [256, 512, 1024]
    # Only params[max_tokens] varies; every other config field is held at base.
    for c in cands:
        assert c.axis == "params"
        assert c.config.system_prompt == "ORIGINAL PROMPT"
        assert c.config.model == "base-model"
        assert c.config.retrieval_ref == "base-snap"


def test_params_axis_preserves_base_params() -> None:
    # The base carries other params keys; overriding one must hold the rest.
    base = _base().with_overrides(params={"temperature": 0, "top_p": 0.9, "seed": 7})
    cands = ParamsAxis("top_p", [0.5, 0.8]).candidates(base)
    for c in cands:
        # Untouched base entries survive on every candidate.
        assert c.config.params["temperature"] == 0
        assert c.config.params["seed"] == 7
    assert [c.config.params["top_p"] for c in cands] == [0.5, 0.8]
    # The base config itself is never mutated (single-axis isolation).
    assert base.params["top_p"] == 0.9


def test_params_axis_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        ParamsAxis("max_tokens", [])


def test_params_axis_rejects_temperature() -> None:
    # replay.forced_determinism_params() forces temperature=0 for every candidate, so a
    # ParamsAxis over it would be a silent no-op; the constructor must reject it by name.
    with pytest.raises(ValueError, match="temperature"):
        ParamsAxis("temperature", [0.0, 0.7, 1.0])


def test_prompt_git_axis_skips_commit_where_file_absent(tmp_path: Path) -> None:
    # A commit that DELETES the file should be skipped (no candidate for it), and the
    # surviving candidates re-numbered contiguously.
    import git

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")
    target = repo_dir / "system.txt"

    target.write_text("v0", encoding="utf-8")
    repo.index.add(["system.txt"])
    repo.index.commit("add v0")

    target.write_text("v1", encoding="utf-8")
    repo.index.add(["system.txt"])
    repo.index.commit("edit v1")

    # Delete the file in a commit that still "touches" the path.
    repo.index.remove(["system.txt"], working_tree=True)
    repo.index.commit("delete system.txt")

    cands = PromptGitAxis(repo_dir, "system.txt").candidates(_base())
    prompts = [c.config.system_prompt for c in cands]
    assert "v0" in prompts
    assert "v1" in prompts
    # The deletion commit produced no candidate; order is contiguous.
    assert [c.order for c in cands] == list(range(len(cands)))
