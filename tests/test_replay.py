"""Tests for deterministic replay: occurrence keying, forced temperature, divergence."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from agentbisect.agent.fake import FakeAgent
from agentbisect.bundle import make_bundle
from agentbisect.mock_tools import DivergencePolicy
from agentbisect.replay import ReplayTemperatureWarning, replay
from agentbisect.types import AgentConfig, ToolStep, Trace


def _counting_executor() -> Any:
    """Executor that returns a fresh, increasing value per call (so each call differs)."""
    state = {"n": 0}

    def _run(tool: str, args: dict[str, Any]) -> Any:
        state["n"] += 1
        return f"{tool}#{state['n']}"

    return _run


def _capture(agent: FakeAgent, config: AgentConfig, executor: Any) -> Trace:
    from agentbisect.capture import capture

    return capture(agent, config, executor, label="t").trace


def test_replay_reproduces_recorded_trace_when_unchanged(fake_agent: FakeAgent) -> None:
    config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [
                {"tool": "search", "args": {"q": "a"}},
                {"llm": "answer"},
            ],
            "final": "FINAL",
        },
    )
    recorded = _capture(fake_agent, config, _counting_executor())
    result = replay(fake_agent, config, recorded)

    assert not result.diverged
    assert result.trace.final_output == "FINAL"
    assert result.trace.tool_steps()[0].output == recorded.tool_steps()[0].output


def test_occurrence_keying_serves_kth_recorded_output(fake_agent: FakeAgent) -> None:
    # The same (tool, args) is called 3 times; each must get the 1st/2nd/3rd output.
    program = [
        {"tool": "search", "args": {"q": "same"}},
        {"tool": "search", "args": {"q": "same"}},
        {"tool": "search", "args": {"q": "same"}},
    ]
    config = AgentConfig(system_prompt="p", model="m", params={"program": program})
    recorded = _capture(fake_agent, config, _counting_executor())
    recorded_outputs = [s.output for s in recorded.tool_steps()]
    assert len(set(recorded_outputs)) == 3  # each call recorded a distinct output

    result = replay(fake_agent, config, recorded)
    replayed_outputs = [s.output for s in result.trace.tool_steps()]
    assert replayed_outputs == recorded_outputs  # 1st/2nd/3rd in order, not a shared output


def test_extra_call_diverges(fake_agent: FakeAgent) -> None:
    # Record 1 call; replay a config that makes 2 -> the 2nd has no recorded match.
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "x"}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    more_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [
                {"tool": "search", "args": {"q": "x"}},
                {"tool": "search", "args": {"q": "x"}},  # extra, unmatched
            ]
        },
    )
    result = replay(fake_agent, more_config, recorded, policy=DivergencePolicy.SKIP)
    assert result.diverged is True


def test_reordered_call_diverges(fake_agent: FakeAgent) -> None:
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [
                {"tool": "alpha", "args": {}},
                {"tool": "beta", "args": {}},
            ]
        },
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    # Replay calls a tool ('gamma') never recorded -> divergence.
    diverge_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "gamma", "args": {}}]},
    )
    result = replay(fake_agent, diverge_config, recorded)
    assert result.diverged is True


def test_nearest_policy_sets_substitution_flag(fake_agent: FakeAgent) -> None:
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "recorded"}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    # Same tool, different args -> NEAREST substitutes the recorded output for that tool.
    new_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "DIFFERENT"}}]},
    )
    result = replay(fake_agent, new_config, recorded, policy=DivergencePolicy.NEAREST)
    assert result.has_nearest_substitutions is True
    assert result.diverged is False


def test_nearest_serves_closest_recorded_not_first(fake_agent: FakeAgent) -> None:
    # Several recorded calls of the same tool with increasingly-matching args. The query
    # is closest (fewest differing fields) to the LAST recorded call, so an honest NEAREST
    # must serve that call's output -- never merely candidates[0] (the first recorded).
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [
                {"tool": "search", "args": {"q": "alpha"}},  # dist 3 from query
                {"tool": "search", "args": {"q": "beta", "region": "eu"}},  # dist 2
                {"tool": "search", "args": {"q": "gamma", "region": "eu", "lang": "en"}},  # dist 1
            ]
        },
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())
    recorded_outputs = [s.output for s in recorded.tool_steps()]
    assert len(set(recorded_outputs)) == 3  # each recorded call has a distinct output
    closest_output = recorded_outputs[2]  # the third call is the genuinely closest

    # Query shares region+lang with the third call, differing only in q -> distance 1.
    query_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [{"tool": "search", "args": {"q": "zeta", "region": "eu", "lang": "en"}}]
        },
    )
    result = replay(fake_agent, query_config, recorded, policy=DivergencePolicy.NEAREST)
    assert result.has_nearest_substitutions is True
    assert result.diverged is False
    served = result.trace.tool_steps()[0].output
    assert served == closest_output  # the closest recorded output, not the first
    assert served != recorded_outputs[0]


def test_nearest_tie_resolves_to_earliest_recorded(fake_agent: FakeAgent) -> None:
    # Two recorded calls are equidistant from the query (each differs in exactly one
    # field). The documented tie-break serves the EARLIEST recorded call.
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={
            "program": [
                {"tool": "search", "args": {"q": "x", "region": "us"}},  # differs in region
                {"tool": "search", "args": {"q": "y", "region": "eu"}},  # differs in q
            ]
        },
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())
    recorded_outputs = [s.output for s in recorded.tool_steps()]

    # Query (q="x", region="eu") is distance 1 from BOTH recorded calls -> tie.
    query_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "x", "region": "eu"}}]},
    )
    result = replay(fake_agent, query_config, recorded, policy=DivergencePolicy.NEAREST)
    served = result.trace.tool_steps()[0].output
    assert served == recorded_outputs[0]  # earliest recorded wins the tie


def test_passthrough_policy_reexecutes_and_flags(fake_agent: FakeAgent) -> None:
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "recorded"}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    new_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "search", "args": {"q": "live"}}]},
    )

    def live(tool: str, args: dict[str, Any]) -> Any:
        return "LIVE_RESULT"

    result = replay(
        fake_agent,
        new_config,
        recorded,
        policy=DivergencePolicy.PASSTHROUGH,
        passthrough_executor=live,
    )
    assert result.used_passthrough is True
    assert result.diverged is False
    assert result.trace.tool_steps()[0].output == "LIVE_RESULT"


def test_replay_forces_temperature_zero_and_warns(fake_agent: FakeAgent) -> None:
    config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"temperature": 0.7, "program": [{"llm": "hi"}], "final": "F"},
    )
    recorded = _capture(fake_agent, config, _counting_executor())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay(fake_agent, config, recorded)

    msgs = [w for w in caught if issubclass(w.category, ReplayTemperatureWarning)]
    assert len(msgs) == 1
    assert "temperature=0" in str(msgs[0].message)


def test_replay_no_warning_when_temperature_zero(fake_agent: FakeAgent) -> None:
    config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"temperature": 0, "program": [{"llm": "hi"}], "final": "F"},
    )
    recorded = _capture(fake_agent, config, _counting_executor())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay(fake_agent, config, recorded)
    assert not [w for w in caught if issubclass(w.category, ReplayTemperatureWarning)]


def test_args_canonicalization_matches_regardless_of_key_order(fake_agent: FakeAgent) -> None:
    # Recorded args in one key order; replayed in another -> still an exact match.
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"a": 1, "b": 2}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    reordered = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"b": 2, "a": 1}}]},
    )
    result = replay(fake_agent, reordered, recorded)
    assert result.diverged is False
    assert result.trace.tool_steps()[0].output == recorded.tool_steps()[0].output


def test_args_canonicalization_float_repr(fake_agent: FakeAgent) -> None:
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"x": 0.1}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())

    # 0.1 vs 0.1000000000001 collapse under .9g canonicalization -> exact match.
    near = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"x": 0.1000000000001}}]},
    )
    result = replay(fake_agent, near, recorded)
    assert result.diverged is False


def test_recorded_trace_has_occurrence_indices(fake_agent: FakeAgent) -> None:
    program = [
        {"tool": "f", "args": {"q": "k"}},
        {"tool": "f", "args": {"q": "k"}},
    ]
    config = AgentConfig(system_prompt="p", model="m", params={"program": program})
    recorded = _capture(fake_agent, config, _counting_executor())
    steps = recorded.tool_steps()
    assert [s.occurrence for s in steps] == [0, 1]
    assert isinstance(steps[0], ToolStep)


def test_passthrough_without_executor_diverges(fake_agent: FakeAgent) -> None:
    rec_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"q": "r"}}]},
    )
    recorded = _capture(fake_agent, rec_config, _counting_executor())
    new_config = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "f", "args": {"q": "MISS"}}]},
    )
    # PASSTHROUGH with no executor falls back to a flagged divergence.
    result = replay(fake_agent, new_config, recorded, policy=DivergencePolicy.PASSTHROUGH)
    assert result.diverged is True


def test_make_bundle_label_is_caller_supplied() -> None:
    bundle = make_bundle(
        config=AgentConfig(system_prompt="p", model="m"),
        trace=Trace(final_output="x"),
        label="my-label",
    )
    assert bundle.label == "my-label"


@pytest.mark.parametrize("temp", [0.0, 0])
def test_zero_temperature_variants_do_not_warn(fake_agent: FakeAgent, temp: float) -> None:
    config = AgentConfig(
        system_prompt="p", model="m", params={"temperature": temp, "program": [{"llm": "x"}]}
    )
    recorded = _capture(fake_agent, config, _counting_executor())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay(fake_agent, config, recorded)
    assert not [w for w in caught if issubclass(w.category, ReplayTemperatureWarning)]
