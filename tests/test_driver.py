"""Tests for the driver: quarantine rule and end-to-end bisection with fakes."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from agentbisect.agent.fake import FakeAgent
from agentbisect.agent.tools import ToolProvider
from agentbisect.bisect import UntestableEndpointError
from agentbisect.driver import ReplayDivergedWarning, make_verdict_fn, run_bisection
from agentbisect.llm.backends import FakeLLM
from agentbisect.mock_tools import DivergencePolicy
from agentbisect.oracle import AssertionOracle, BackendError, FakeOracle, LLMJudge
from agentbisect.timeout import CandidateTimeout, expire_after
from agentbisect.types import AgentConfig, Candidate, RunBundle, Trace, Verdict


def _counting_executor() -> Callable[[str, dict[str, Any]], Any]:
    state = {"n": 0}

    def _run(tool: str, args: dict[str, Any]) -> Any:
        state["n"] += 1
        return f"{tool}#{state['n']}"

    return _run


def _capture(agent: FakeAgent, config: AgentConfig) -> RunBundle:
    from agentbisect.capture import capture

    return capture(agent, config, _counting_executor(), label="t")


# ----------------------------------------------------------------- quarantine rule


def test_backend_error_from_judge_names_the_candidate(fake_agent: FakeAgent) -> None:
    """A raising judge backend is re-raised with the failing candidate's ref, not SKIP."""
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)

    def boom(system: str, user: str) -> str:
        raise RuntimeError("expired api key")

    oracle = LLMJudge(FakeLLM(boom))
    verdict_fn = make_verdict_fn(fake_agent, bundle, oracle)
    candidate = Candidate(axis="model", ref="m0", order=0, config=base)
    with pytest.raises(BackendError, match="m0") as caught:
        verdict_fn(candidate)
    assert caught.value.candidate_ref == "m0"
    assert caught.value.candidate_order == 0


def test_diverged_replay_is_quarantined_as_skip(fake_agent: FakeAgent) -> None:
    # Capture a run with one tool call.
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)

    # An oracle that would say BAD if it ever ran -- but the candidate diverges, so the
    # quarantine must force SKIP regardless of the oracle's opinion.
    oracle = FakeOracle(default=Verdict.BAD)
    verdict_fn = make_verdict_fn(fake_agent, bundle, oracle, policy=DivergencePolicy.SKIP)

    diverging = Candidate(
        axis="model",
        ref="m1",
        order=1,
        config=base.with_overrides(
            model="m1",
            params={"program": [{"tool": "search", "args": {"q": "DIFFERENT"}}], "final": "OUT"},
        ),
    )
    assert verdict_fn(diverging) is Verdict.SKIP


def test_nearest_substitution_is_quarantined_as_skip(fake_agent: FakeAgent) -> None:
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)
    oracle = FakeOracle(default=Verdict.GOOD)  # would say GOOD; quarantine overrides
    verdict_fn = make_verdict_fn(fake_agent, bundle, oracle, policy=DivergencePolicy.NEAREST)

    subbed = Candidate(
        axis="model",
        ref="m1",
        order=1,
        config=base.with_overrides(
            params={"program": [{"tool": "search", "args": {"q": "OTHER"}}], "final": "OUT"},
        ),
    )
    assert verdict_fn(subbed) is Verdict.SKIP


def test_timeout_overrun_is_quarantined_as_skip(fake_agent: FakeAgent) -> None:
    """A runner that 'sleeps' past the timeout (fake clock) yields skip, not bad."""
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)
    oracle = FakeOracle(default=Verdict.BAD)  # would say BAD; timeout must win
    verdict_fn = make_verdict_fn(
        fake_agent, bundle, oracle, timeout=0.5, timeout_impl=expire_after(10.0)
    )
    candidate = Candidate(axis="model", ref="m0", order=0, config=base)
    assert verdict_fn(candidate) is Verdict.SKIP


def test_timeout_overrun_never_yields_first_bad(fake_agent: FakeAgent) -> None:
    """Every candidate overrunning the deadline is untestable, never a culprit."""
    good_prompt = "You are support. Always state the refund policy."
    bundle = _capture(fake_agent, _regression_config("m0", good_prompt))
    candidates = [
        Candidate(
            axis="model",
            ref=f"m{i}",
            order=i,
            config=_regression_config(f"m{i}", good_prompt),
        )
        for i in range(3)
    ]
    oracle = AssertionOracle(lambda t, b: True)
    with pytest.raises(UntestableEndpointError):
        run_bisection(
            fake_agent,
            bundle,
            candidates,
            oracle,
            timeout=0.25,
            timeout_impl=expire_after(5.0),
        )


def test_interior_timeout_is_ambiguous_never_first_bad(fake_agent: FakeAgent) -> None:
    """Endpoints finish; interior overruns are skip -> ambiguous, not first_bad."""
    good_prompt = "You are support. Always state the refund policy."
    bundle = _capture(fake_agent, _regression_config("m0", good_prompt))
    prompts = [
        good_prompt,
        good_prompt,
        "You are support. Be brief.",
        "You are support. Be brief.",
        "You are support. Be brief.",
    ]
    candidates = [
        Candidate(
            axis="model",
            ref=f"m{i}",
            order=i,
            config=_regression_config(f"m{i}", prompts[i]),
        )
        for i in range(5)
    ]
    calls = {"n": 0}

    def impl(fn: Any, seconds: float) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            return fn()
        raise CandidateTimeout(f"candidate timed out after {seconds}s")

    oracle = AssertionOracle(lambda t, b: t.final_output == "refund_clause=yes")
    outcome = run_bisection(fake_agent, bundle, candidates, oracle, timeout=0.5, timeout_impl=impl)
    assert outcome.result.first_bad is None
    assert outcome.result.is_ambiguous


def test_fast_runner_with_timeout_is_unaffected(fake_agent: FakeAgent) -> None:
    """A runner that finishes under the deadline still isolates first_bad."""
    good_prompt = "You are support. Always state the refund policy."
    bundle = _capture(fake_agent, _regression_config("m0", good_prompt))
    prompts = [good_prompt, good_prompt, "You are support. Be brief."]
    candidates = [
        Candidate(
            axis="model",
            ref=f"m{i}",
            order=i,
            config=_regression_config(f"m{i}", prompts[i]),
        )
        for i in range(3)
    ]
    oracle = AssertionOracle(lambda t, b: t.final_output == "refund_clause=yes")
    outcome = run_bisection(
        fake_agent,
        bundle,
        candidates,
        oracle,
        timeout=5.0,
        timeout_impl=expire_after(0.0),
    )
    assert outcome.result.first_bad is not None
    assert outcome.result.first_bad.order == 2


def test_matching_replay_uses_oracle_verdict(fake_agent: FakeAgent) -> None:
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)
    oracle = FakeOracle(by_output={"OUT": Verdict.BAD})
    verdict_fn = make_verdict_fn(fake_agent, bundle, oracle)

    same = Candidate(axis="model", ref="m0", order=0, config=base)
    assert verdict_fn(same) is Verdict.BAD


# ------------------------------------------------------------ end-to-end bisection


def _regression_config(model: str, prompt: str) -> AgentConfig:
    # The FakeAgent's final output encodes whether the prompt still mentions 'refund'.
    return AgentConfig(
        system_prompt=prompt,
        model=model,
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund_clause={prompt_has:refund}",
        },
    )


def test_end_to_end_bisection_finds_first_bad(fake_agent: FakeAgent) -> None:
    # Capture the OLD (good) run that still mentions the refund clause.
    good_prompt = "You are support. Always state the refund policy."
    bundle = _capture(fake_agent, _regression_config("m0", good_prompt))

    # Build a model axis where models m0..m2 keep the refund clause and m3..m4 drop it.
    # We simulate the regression on the PROMPT carried per-candidate via params.
    prompts = [
        good_prompt,  # m0 good
        good_prompt,  # m1 good
        good_prompt,  # m2 good
        "You are support. Be brief.",  # m3 bad (no refund)
        "You are support. Be brief.",  # m4 bad
    ]
    candidates = [
        Candidate(
            axis="model",
            ref=f"m{i}",
            order=i,
            config=_regression_config(f"m{i}", prompts[i]),
        )
        for i in range(5)
    ]

    # Oracle: GOOD iff the final output reports the refund clause present.
    oracle = AssertionOracle(lambda t, b: t.final_output == "refund_clause=yes")

    outcome = run_bisection(fake_agent, bundle, candidates, oracle)
    assert outcome.result.first_bad is not None
    assert outcome.result.first_bad.order == 3
    assert outcome.result.last_good is not None
    assert outcome.result.last_good.order == 2

    # Report artifacts are populated for a single first-bad result.
    assert outcome.behavioral_diff is not None
    assert outcome.behavioral_diff.final_output_changed
    assert outcome.minimal_repro is not None
    assert len(outcome.minimal_repro.steps) >= 1


def test_run_bisection_uses_model_list_axis(fake_agent: FakeAgent) -> None:
    # The ModelListAxis overrides only the model; to make the FakeAgent's output depend
    # on the model we encode the model name into the system prompt and final output.
    base = AgentConfig(
        system_prompt="model=m0",
        model="m0",
        params={"program": [], "final": "out_for={prompt_has:m2}"},
    )
    bundle = _capture(fake_agent, base)

    # Build candidates whose prompt names the model, so m2 produces a distinct output.
    candidates = [
        Candidate(
            axis="model",
            ref=f"m{i}",
            order=i,
            config=base.with_overrides(model=f"m{i}", system_prompt=f"model=m{i}"),
        )
        for i in range(3)
    ]

    # m0,m1 -> "out_for=no" (good); m2 -> "out_for=yes" (bad).
    oracle = FakeOracle(
        by_output={"out_for=no": Verdict.GOOD, "out_for=yes": Verdict.BAD},
    )
    outcome = run_bisection(fake_agent, bundle, candidates, oracle)
    assert outcome.result.first_bad is not None
    assert outcome.result.first_bad.order == 2


def test_passthrough_surfaced_in_outcome(fake_agent: FakeAgent) -> None:
    base = AgentConfig(
        system_prompt="p",
        model="m0",
        params={"program": [{"tool": "search", "args": {"q": "x"}}], "final": "OUT"},
    )
    bundle = _capture(fake_agent, base)

    def live(tool: str, args: dict[str, Any]) -> Any:
        return "LIVE"

    # A 2-candidate axis: endpoint 0 matches (good), endpoint 1 passes through (bad).
    candidates = [
        Candidate(axis="model", ref="m0", order=0, config=base),
        Candidate(
            axis="model",
            ref="m1",
            order=1,
            config=base.with_overrides(
                params={"program": [{"tool": "search", "args": {"q": "MISS"}}], "final": "OUT2"},
            ),
        ),
    ]
    oracle = FakeOracle(by_output={"OUT": Verdict.GOOD, "OUT2": Verdict.BAD})

    from agentbisect.bisect import bisect

    verdict_fn = make_verdict_fn(
        fake_agent,
        bundle,
        oracle,
        policy=DivergencePolicy.PASSTHROUGH,
        passthrough_executor=live,
        on_passthrough=lambda c, r: None,
    )
    result = bisect(candidates, verdict_fn)
    assert result.first_bad is not None
    assert result.first_bad.order == 1


# ---------------------------------------------- post-bisect artifact replay quarantine


class _DivergeOnReplayRunner:
    """A runner that reproduces faithfully on first replay but diverges on a re-replay.

    It delegates to :class:`FakeAgent` for the first run of any config (so the bisect can
    resolve a correct ``first_bad``), but on a *second* run of a designated config it
    appends an unmatched tool call -- the divergence the post-bisect artifact replay must
    quarantine instead of forwarding a corrupt trace to diff()/minimize().
    """

    def __init__(self, diverge_fingerprint: str) -> None:
        self._inner = FakeAgent()
        self._diverge_fp = diverge_fingerprint
        self._runs: dict[str, int] = {}

    def run(self, config: AgentConfig, tools: ToolProvider) -> Trace:
        fp = config.fingerprint()
        count = self._runs.get(fp, 0)
        self._runs[fp] = count + 1
        if fp == self._diverge_fp and count >= 1:
            # Second+ replay of the bad config: make an unmatched call -> divergence.
            program = [*config.params.get("program", []), {"tool": "kb", "args": {"q": "DIVERGED"}}]
            config = config.with_overrides(params={**config.params, "program": program})
        return self._inner.run(config, tools)


def test_post_bisect_replay_divergence_yields_unavailable_not_corrupt() -> None:
    # Capture a good run with a single matched tool call. ``temperature=0`` is set so the
    # replay's forced-determinism pass is a no-op and the fingerprint stays stable across
    # capture and replay (the divergence is keyed on that fingerprint).
    good_cfg = AgentConfig(
        system_prompt="You always state the refund policy.",
        model="m0",
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund={prompt_has:refund}",
            "temperature": 0,
        },
    )
    bad_cfg = good_cfg.with_overrides(
        system_prompt="Be brief.",  # drops 'refund' -> final output flips to BAD
        model="m1",
    )

    runner = _DivergeOnReplayRunner(diverge_fingerprint=bad_cfg.fingerprint())
    bundle = _capture(runner, good_cfg)

    candidates = [
        Candidate(axis="model", ref="m0", order=0, config=good_cfg),
        Candidate(axis="model", ref="m1", order=1, config=bad_cfg),
    ]
    # GOOD iff the final output reports the refund clause present.
    oracle = AssertionOracle(lambda t, b: t.final_output == "refund=yes")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = run_bisection(runner, bundle, candidates, oracle)

    # first_bad is still correctly isolated despite the post-bisect replay diverging.
    assert outcome.result.first_bad is not None
    assert outcome.result.first_bad.order == 1
    assert outcome.result.last_good is not None
    assert outcome.result.last_good.order == 0

    # Artifacts are a clean "unavailable" state, NOT a corrupt/empty minimal repro.
    assert outcome.minimal_repro is None
    assert outcome.behavioral_diff is None
    assert outcome.artifacts_unavailable is not None
    assert "replay diverged" in outcome.artifacts_unavailable

    # The non-determinism was surfaced (never silent).
    diverged_warnings = [w for w in caught if issubclass(w.category, ReplayDivergedWarning)]
    assert len(diverged_warnings) == 1


def test_clean_replay_still_builds_artifacts() -> None:
    # Control: a runner that reproduces deterministically yields real artifacts and no
    # "unavailable" marker (guards against the quarantine firing spuriously).
    agent = FakeAgent()
    good_cfg = AgentConfig(
        system_prompt="You always state the refund policy.",
        model="m0",
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund={prompt_has:refund}",
        },
    )
    bad_cfg = good_cfg.with_overrides(system_prompt="Be brief.", model="m1")
    bundle = _capture(agent, good_cfg)
    candidates = [
        Candidate(axis="model", ref="m0", order=0, config=good_cfg),
        Candidate(axis="model", ref="m1", order=1, config=bad_cfg),
    ]
    oracle = AssertionOracle(lambda t, b: t.final_output == "refund=yes")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = run_bisection(agent, bundle, candidates, oracle)

    assert outcome.result.first_bad is not None
    assert outcome.result.first_bad.order == 1
    assert outcome.artifacts_unavailable is None
    assert outcome.behavioral_diff is not None
    assert outcome.minimal_repro is not None
    assert not [w for w in caught if issubclass(w.category, ReplayDivergedWarning)]
