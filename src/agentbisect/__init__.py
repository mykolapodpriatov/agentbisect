"""agentbisect -- git-bisect for LLM-agent regressions.

Capture a failing agent run as a versioned bundle (prompt, model id/params, tool
schemas, recorded tool I/O), replay it deterministically by mocking tool calls from the
recording, then binary-search an ordered set of versions (prompt git history, a model
list, tool-schema versions) driven by an LLM-judge or assertion oracle to pinpoint the
first change that broke behavior -- and emit a minimal reproducing trace + a side-by-side
behavioral diff.
"""

from __future__ import annotations

from .axes import ModelListAxis, PromptGitAxis, RetrievalAxis, ToolSchemaAxis
from .bisect import (
    BisectError,
    NonMonotonicError,
    UntestableEndpointError,
    bisect,
)
from .bundle import (
    BundleIntegrityError,
    BundleVersionError,
    load_bundle,
    make_bundle,
    save_bundle,
)
from .capture import capture
from .diff import BehavioralDiff, diff
from .driver import BisectionOutcome, make_verdict_fn, run_bisection
from .minimize import minimize
from .mock_tools import DivergencePolicy
from .oracle import AssertionOracle, FakeOracle, LLMJudge, Oracle
from .replay import ReplayTemperatureWarning, replay
from .types import (
    AgentConfig,
    BisectResult,
    Candidate,
    LlmStep,
    ReplayResult,
    RunBundle,
    Step,
    ToolSchema,
    ToolStep,
    Trace,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "AgentConfig",
    "ToolSchema",
    "Step",
    "LlmStep",
    "ToolStep",
    "Trace",
    "ReplayResult",
    "RunBundle",
    "Verdict",
    "Candidate",
    "BisectResult",
    # bundle
    "make_bundle",
    "save_bundle",
    "load_bundle",
    "BundleVersionError",
    "BundleIntegrityError",
    # capture / replay
    "capture",
    "replay",
    "ReplayTemperatureWarning",
    "DivergencePolicy",
    # bisect
    "bisect",
    "BisectError",
    "UntestableEndpointError",
    "NonMonotonicError",
    # driver
    "make_verdict_fn",
    "run_bisection",
    "BisectionOutcome",
    # oracle
    "Oracle",
    "AssertionOracle",
    "FakeOracle",
    "LLMJudge",
    # axes
    "ModelListAxis",
    "PromptGitAxis",
    "ToolSchemaAxis",
    "RetrievalAxis",
    # diff / minimize
    "diff",
    "BehavioralDiff",
    "minimize",
]
