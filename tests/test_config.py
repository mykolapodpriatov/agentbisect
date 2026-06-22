"""Tests for project-config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentbisect.config import ConfigError, load_project_config


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_project_config(tmp_path / "missing.py")


def test_load_invalid_module_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(bad)


def test_missing_hook_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "p.py"
    cfg.write_text("x = 1\n", encoding="utf-8")
    project = load_project_config(cfg)
    with pytest.raises(ConfigError):
        project.runner()


def test_hooks_resolve(tmp_path: Path) -> None:
    cfg = tmp_path / "p.py"
    cfg.write_text(
        "from agentbisect.agent.fake import FakeAgent\n"
        "from agentbisect.oracle import FakeOracle\n"
        "from agentbisect.types import AgentConfig\n"
        "def runner():\n    return FakeAgent()\n"
        "def oracle():\n    return FakeOracle()\n"
        "def config():\n    return AgentConfig(system_prompt='p', model='m')\n"
        "def tool_executor():\n    return lambda t, a: 'x'\n",
        encoding="utf-8",
    )
    project = load_project_config(cfg)
    assert project.runner() is not None
    assert project.oracle() is not None
    assert project.config().model == "m"
    assert project.tool_executor()("t", {}) == "x"
