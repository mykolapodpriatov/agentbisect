"""Project configuration: how to construct the user's runner, oracle, and tools.

A project config is a plain Python file that defines any of these module-level callables:

* ``runner() -> AgentRunner`` (required) -- builds the agent adapter.
* ``oracle() -> Oracle`` (required for ``bisect``) -- builds the good/bad/skip judge.
* ``tool_executor() -> Callable[[str, dict], Any]`` (required for ``capture``) -- performs
  real tool calls during capture.
* ``config() -> AgentConfig`` (required for ``capture``) -- the config to capture.

Loading a plain Python file (rather than a declarative format) is deliberate: a user's
runner/oracle are arbitrary code, and this keeps agentbisect a thin adapter rather than a
framework. The file is imported in isolation via :mod:`importlib`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .agent import AgentRunner
from .oracle import Oracle
from .types import AgentConfig

__all__ = ["ConfigError", "ProjectConfig", "load_project_config"]


class ConfigError(Exception):
    """Raised when a project config file is missing required hooks or is invalid."""


class ProjectConfig:
    """A loaded project config exposing lazily-constructed runner/oracle/etc."""

    def __init__(self, module: object, source: Path) -> None:
        self._module = module
        self._source = source

    def _hook(self, name: str) -> Callable[[], Any]:
        fn = getattr(self._module, name, None)
        if fn is None or not callable(fn):
            raise ConfigError(
                f"project config {self._source} does not define a callable {name!r}()"
            )
        return cast(Callable[[], Any], fn)

    def runner(self) -> AgentRunner:
        """Construct the user's agent runner."""
        return cast(AgentRunner, self._hook("runner")())

    def oracle(self) -> Oracle:
        """Construct the good/bad/skip oracle."""
        return cast(Oracle, self._hook("oracle")())

    def tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        """Construct the real tool executor used during capture."""
        return cast(Callable[[str, dict[str, Any]], Any], self._hook("tool_executor")())

    def config(self) -> AgentConfig:
        """Construct the agent config to capture."""
        return cast(AgentConfig, self._hook("config")())


def load_project_config(path: str | Path) -> ProjectConfig:
    """Import a project config file and return a :class:`ProjectConfig`.

    Raises
    ------
    ConfigError
        If the file does not exist or cannot be imported.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise ConfigError(f"project config not found: {source}")

    module_name = f"_agentbisect_project_{abs(hash(str(source)))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib edge case
        raise ConfigError(f"cannot import project config: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"error importing project config {source}: {exc}") from exc
    return ProjectConfig(module, source)
