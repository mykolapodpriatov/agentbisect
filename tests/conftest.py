"""Shared, fully-offline fixtures for the agentbisect test suite.

Everything here is deterministic and network-free: a :class:`FakeAgent` driven by a tiny
embedded program, recorded tool outputs, a scripted :class:`FakeOracle`, and a temporary
git repo (with a fixed identity) for the prompt-history axis.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentbisect.agent.fake import FakeAgent
from agentbisect.bundle import make_bundle
from agentbisect.types import AgentConfig, RunBundle


@pytest.fixture
def fake_agent() -> FakeAgent:
    """A deterministic, network-free agent."""
    return FakeAgent()


def _echo_executor(tool: str, args: dict[str, Any]) -> Any:
    """Deterministic tool executor: echoes a stable string per call."""
    return f"{tool}:{sorted(args.items())}"


@pytest.fixture
def echo_executor() -> Callable[[str, dict[str, Any]], Any]:
    """A deterministic tool executor for capture."""
    return _echo_executor


def make_config(
    *,
    system_prompt: str = "base prompt",
    model: str = "model-a",
    program: list[dict[str, Any]] | None = None,
    final: str | None = None,
    params: dict[str, Any] | None = None,
) -> AgentConfig:
    """Build an :class:`AgentConfig` whose FakeAgent program is embedded in params."""
    p: dict[str, Any] = dict(params or {})
    if program is not None:
        p["program"] = program
    if final is not None:
        p["final"] = final
    return AgentConfig(system_prompt=system_prompt, model=model, params=p)


@pytest.fixture
def config_factory() -> Callable[..., AgentConfig]:
    """Expose :func:`make_config` as a fixture."""
    return make_config


def make_bundle_from(
    agent: FakeAgent,
    config: AgentConfig,
    executor: Callable[[str, dict[str, Any]], Any],
    *,
    label: str = "test",
) -> RunBundle:
    """Capture ``agent`` under ``config`` into a bundle (deterministically)."""
    from agentbisect.capture import capture

    return capture(agent, config, executor, label=label)


@pytest.fixture
def bundle_factory(
    fake_agent: FakeAgent,
    echo_executor: Callable[[str, dict[str, Any]], Any],
) -> Callable[..., RunBundle]:
    """Return a helper that captures a bundle from a config."""

    def _factory(config: AgentConfig, *, label: str = "test") -> RunBundle:
        return make_bundle_from(fake_agent, config, echo_executor, label=label)

    return _factory


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Callable[[str, list[str]], Path]:
    """Create a temp git repo and commit successive versions of a file.

    Returns a factory ``make(path, versions) -> repo_dir`` that writes each version of
    ``path`` as its own commit (old -> new). Git identity is set locally so commits work
    in CI without a global config.
    """
    import git

    def _make(rel_path: str, versions: list[str]) -> Path:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(exist_ok=True)
        repo = git.Repo.init(repo_dir)
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "Test User")
            cw.set_value("user", "email", "test@example.com")
        target = repo_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        for i, content in enumerate(versions):
            target.write_text(content, encoding="utf-8")
            repo.index.add([rel_path])
            repo.index.commit(f"version {i}")
        return repo_dir

    return _make


@pytest.fixture
def make_bundle_helper() -> Callable[..., RunBundle]:
    """Expose :func:`agentbisect.bundle.make_bundle` for direct trace construction."""
    return make_bundle
