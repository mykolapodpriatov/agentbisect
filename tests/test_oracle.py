"""Tests for oracles: AssertionOracle, FakeOracle, and the cached LLMJudge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbisect.llm.backends import FakeLLM
from agentbisect.oracle import (
    AssertionOracle,
    FakeOracle,
    JudgeCache,
    LLMJudge,
)
from agentbisect.types import AgentConfig, LlmStep, RunBundle, Trace, Verdict


def _bundle(final: str = "out", label: str = "case") -> RunBundle:
    return RunBundle(
        version=1,
        config=AgentConfig(system_prompt="p", model="m"),
        trace=Trace(final_output=final),
        label=label,
    )


def _trace(final: str = "out") -> Trace:
    return Trace(steps=(LlmStep(index=0, role="assistant", content="x"),), final_output=final)


# ------------------------------------------------------------------ AssertionOracle


def test_assertion_oracle_good_and_bad() -> None:
    good = AssertionOracle(lambda t, b: "refund" in t.final_output)
    assert good.judge(_trace("refund: 30 days"), _bundle()) is Verdict.GOOD
    assert good.judge(_trace("no policy"), _bundle()) is Verdict.BAD


def test_assertion_oracle_undecidable_is_skip() -> None:
    def predicate(t: Trace, b: RunBundle) -> bool:
        raise AssertionOracle.Undecidable

    oracle = AssertionOracle(predicate)
    assert oracle.judge(_trace(), _bundle()) is Verdict.SKIP


# ----------------------------------------------------------------------- FakeOracle


def test_fake_oracle_by_output() -> None:
    oracle = FakeOracle(by_output={"good-out": Verdict.GOOD, "bad-out": Verdict.BAD})
    assert oracle.judge(_trace("good-out"), _bundle()) is Verdict.GOOD
    assert oracle.judge(_trace("bad-out"), _bundle()) is Verdict.BAD


def test_fake_oracle_sequence() -> None:
    oracle = FakeOracle(sequence=[Verdict.GOOD, Verdict.BAD, Verdict.SKIP])
    assert oracle.judge(_trace(), _bundle()) is Verdict.GOOD
    assert oracle.judge(_trace(), _bundle()) is Verdict.BAD
    assert oracle.judge(_trace(), _bundle()) is Verdict.SKIP
    # Past the end it repeats the last entry (stable).
    assert oracle.judge(_trace(), _bundle()) is Verdict.SKIP


def test_fake_oracle_default_is_skip() -> None:
    oracle = FakeOracle(by_output={"x": Verdict.GOOD})
    assert oracle.judge(_trace("unknown"), _bundle()) is Verdict.SKIP


def test_fake_oracle_flaky_by_ref_alternates() -> None:
    bundle = _bundle()
    fp = bundle.config.fingerprint()
    oracle = FakeOracle(by_ref={fp: [Verdict.GOOD, Verdict.BAD]})
    assert oracle.judge(_trace(), bundle) is Verdict.GOOD
    assert oracle.judge(_trace(), bundle) is Verdict.BAD
    assert oracle.judge(_trace(), bundle) is Verdict.GOOD


# ------------------------------------------------------------------------- LLMJudge


def test_llm_judge_parses_and_caches(tmp_path: Path) -> None:
    calls = {"n": 0}

    def responder(system: str, user: str) -> str:
        calls["n"] += 1
        return "BAD"

    backend = FakeLLM(responder, configured_model="judge-x")
    cache = JudgeCache(tmp_path / "judge.json")
    judge = LLMJudge(backend, cache=cache)

    t = _trace("answer")
    assert judge.judge(t, _bundle()) is Verdict.BAD
    assert calls["n"] == 1
    # Second judgement of the same trace is served from cache (no extra backend call).
    assert judge.judge(t, _bundle()) is Verdict.BAD
    assert calls["n"] == 1


def test_llm_judge_undecidable_is_skip() -> None:
    judge = LLMJudge(FakeLLM(lambda s, u: "I am not sure"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP


def test_llm_judge_good_verdict() -> None:
    judge = LLMJudge(FakeLLM(lambda s, u: "GOOD"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.GOOD


# ----------------------------------------------------- verdict parsing (exact one word)


def test_parse_verdict_not_bad_is_skip_not_bad() -> None:
    # "not bad" must NOT be misclassified as BAD via a substring match; it is undecidable.
    judge = LLMJudge(FakeLLM(lambda s, u: "not bad"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP


def test_parse_verdict_good_enough_is_skip_not_good() -> None:
    # "good enough" must NOT be misclassified as GOOD via a substring match.
    judge = LLMJudge(FakeLLM(lambda s, u: "good enough"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP


def test_parse_verdict_exact_bad_is_bad() -> None:
    # An exact one-word verdict (case-insensitive, stripped) is honored.
    judge = LLMJudge(FakeLLM(lambda s, u: "  Bad\n"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.BAD


def test_parse_verdict_exact_skip_word_is_skip() -> None:
    judge = LLMJudge(FakeLLM(lambda s, u: "SKIP"), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP


# --- verdict parsing: real models wrap the one word in punctuation/markdown -----------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("GOOD.", Verdict.GOOD),  # trailing period
        ("bad,", Verdict.BAD),  # trailing comma
        ("**good**", Verdict.GOOD),  # markdown bold
        ("`bad`", Verdict.BAD),  # markdown inline code
        ('"skip"', Verdict.SKIP),  # double-quoted
        ("'good'", Verdict.GOOD),  # single-quoted
        ("  BAD  ", Verdict.BAD),  # surrounding whitespace + caps
        ("good!", Verdict.GOOD),  # exclamation
        ("Bad?", Verdict.BAD),  # question mark + caps
        ("skip:", Verdict.SKIP),  # colon
    ],
)
def test_parse_verdict_unwraps_punctuation_and_markdown(answer: str, expected: Verdict) -> None:
    """A single one-word verdict wrapped in common punctuation/markdown is honored."""
    judge = LLMJudge(FakeLLM(lambda s, u: answer), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is expected


@pytest.mark.parametrize(
    "answer",
    [
        "not bad",  # multi-word: must NOT substring-match BAD
        "good enough",  # multi-word: must NOT substring-match GOOD
        "goodbad",  # glued token: not an exact verdict word
        "**not bad**",  # wrapped multi-word phrase stays undecidable
        "the answer is good",  # verdict embedded in a sentence
        "",  # empty
        "   ",  # whitespace only
    ],
)
def test_parse_verdict_non_single_verdict_word_is_skip(answer: str) -> None:
    """Anything that is not a single bare verdict word maps to SKIP (no substring match)."""
    judge = LLMJudge(FakeLLM(lambda s, u: answer), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP


def test_resolved_model_id_alias_single_cache_entry(tmp_path: Path) -> None:
    """The alias and the resolved id must collapse to a single cache entry (no double-judge)."""
    cache_path = tmp_path / "judge.json"

    # Judge #1 is configured with an alias but the backend resolves to a concrete id.
    alias_calls = {"n": 0}

    def alias_responder(system: str, user: str) -> str:
        alias_calls["n"] += 1
        return "BAD"

    alias_backend = FakeLLM(
        alias_responder,
        configured_model="judge-latest",  # moving alias
        resolved_model="judge-2026-06-01",  # concrete id reported after the first call
    )
    judge_alias = LLMJudge(alias_backend, cache=JudgeCache(cache_path))
    t = _trace("answer")
    assert judge_alias.judge(t, _bundle()) is Verdict.BAD
    assert alias_calls["n"] == 1

    # Exactly one entry was written, keyed by the RESOLVED id.
    cache_after_alias = JudgeCache(cache_path)
    assert len(cache_after_alias) == 1

    # Judge #2 is configured directly with the resolved id; it must HIT the same entry.
    resolved_calls = {"n": 0}

    def resolved_responder(system: str, user: str) -> str:
        resolved_calls["n"] += 1
        return "GOOD"  # would differ, proving we hit cache and did NOT call the backend

    resolved_backend = FakeLLM(resolved_responder, configured_model="judge-2026-06-01")
    judge_resolved = LLMJudge(resolved_backend, cache=JudgeCache(cache_path))
    assert judge_resolved.judge(t, _bundle()) is Verdict.BAD  # cached value, not GOOD
    assert resolved_calls["n"] == 0  # backend never called -> single shared cache key

    # Still exactly one cache entry.
    assert len(JudgeCache(cache_path)) == 1


def test_cold_alias_instance_hits_resolved_entry_with_zero_backend_calls(tmp_path: Path) -> None:
    """A FRESH judge configured with a moving alias hits the resolved-id entry, no backend call.

    Instance A (alias) judges, populating both the verdict entry (keyed by the resolved id) and
    the persisted alias->resolved-id map. Instance B is a cold instance (separate object, as a
    new process would be) configured with the SAME alias; its backend is stubbed to FAIL if
    called. B must read A's verdict from cache without ever calling the backend -- proving the
    "alias and resolved id collapse to one entry" claim holds across cold instances.
    """
    cache_path = tmp_path / "judge.json"
    t = _trace("answer")
    bundle = _bundle()

    # --- Instance A: configured with a moving alias, resolves to a concrete id on first call.
    a_calls = {"n": 0}

    def a_responder(system: str, user: str) -> str:
        a_calls["n"] += 1
        return "BAD"

    backend_a = FakeLLM(
        a_responder,
        configured_model="judge-latest",  # moving alias
        resolved_model="judge-2026-06-01",  # concrete id reported after the first call
    )
    judge_a = LLMJudge(backend_a, cache=JudgeCache(cache_path))
    assert judge_a.judge(t, bundle) is Verdict.BAD
    assert a_calls["n"] == 1

    # The cache file now persists the alias->resolved mapping for cold instances to reuse.
    persisted = JudgeCache(cache_path)
    assert persisted.resolved_alias("judge-latest") == "judge-2026-06-01"
    assert len(persisted) == 1  # exactly one verdict entry, keyed by the resolved id

    # --- Instance B: a cold instance with the SAME alias and a backend that FAILS if called.
    def fail_if_called(system: str, user: str) -> str:
        raise AssertionError("backend must not be called: a cold alias instance must hit cache")

    backend_b = FakeLLM(
        fail_if_called,
        configured_model="judge-latest",  # same moving alias, resolved id NOT yet known to B
        resolved_model="judge-2026-06-01",
    )
    judge_b = LLMJudge(backend_b, cache=JudgeCache(cache_path))
    assert judge_b.judge(t, bundle) is Verdict.BAD  # served from the resolved-id cache entry

    # Still exactly one verdict entry: the alias did not create a second, alias-keyed entry.
    assert len(JudgeCache(cache_path)) == 1


def test_cold_alias_instance_hits_resolved_entry_committed_by_resolved_id_judge(
    tmp_path: Path,
) -> None:
    """Even if the entry was first written by a judge configured with the RESOLVED id directly,
    a cold instance configured with the ALIAS still hits it once the alias map is known.

    Here a judge configured with the concrete id writes the verdict, and a separate alias-map
    seeding establishes the alias->resolved mapping; a cold alias instance then resolves the
    alias from the map and reads the verdict with zero backend calls.
    """
    cache_path = tmp_path / "judge.json"
    t = _trace("answer")
    bundle = _bundle()

    # A judge configured directly with the concrete id writes the verdict entry.
    backend_resolved = FakeLLM(lambda s, u: "GOOD", configured_model="judge-2026-06-01")
    judge_resolved = LLMJudge(backend_resolved, cache=JudgeCache(cache_path))
    assert judge_resolved.judge(t, bundle) is Verdict.GOOD

    # Record the alias->resolved mapping (as the first alias-configured judge anywhere would).
    seed = JudgeCache(cache_path)
    seed.set_alias("judge-latest", "judge-2026-06-01")

    # A cold instance with the alias and a backend that FAILS if called hits the entry.
    def fail_if_called(system: str, user: str) -> str:
        raise AssertionError("backend must not be called for a known alias")

    backend_alias = FakeLLM(
        fail_if_called,
        configured_model="judge-latest",
        resolved_model="judge-2026-06-01",
    )
    judge_alias = LLMJudge(backend_alias, cache=JudgeCache(cache_path))
    assert judge_alias.judge(t, bundle) is Verdict.GOOD
    assert len(JudgeCache(cache_path)) == 1


def test_judge_cache_reads_legacy_flat_layout(tmp_path: Path) -> None:
    """A pre-existing flat ``{key: verdict}`` cache file (no sections) is read transparently."""
    cache_path = tmp_path / "judge.json"
    seed_path = tmp_path / "seed.json"
    t = _trace("answer")
    bundle = _bundle()

    # First produce a real entry to learn its exact cache key, then rewrite the file in the
    # legacy flat layout an older version of the tool would have written.
    seed_backend = FakeLLM(lambda s, u: "BAD", configured_model="m")
    seeder = LLMJudge(seed_backend, cache=JudgeCache(seed_path))
    seeder.judge(t, bundle)
    seeded = json.loads(seed_path.read_text(encoding="utf-8"))
    (legacy_key,) = seeded["verdicts"]
    cache_path.write_text(json.dumps({legacy_key: "bad"}, indent=2) + "\n", encoding="utf-8")

    cache = JudgeCache(cache_path)
    assert len(cache) == 1
    assert cache.get(legacy_key) is Verdict.BAD

    # A fresh judge configured with that id reads the legacy verdict with no backend call.
    calls = {"n": 0}

    def responder(system: str, user: str) -> str:
        calls["n"] += 1
        return "GOOD"

    judge = LLMJudge(FakeLLM(responder, configured_model="m"), cache=JudgeCache(cache_path))
    assert judge.judge(t, bundle) is Verdict.BAD
    assert calls["n"] == 0


def test_distinct_rendered_prompts_do_not_collide_in_cache(tmp_path: Path) -> None:
    """Two bundles whose rendered prompts differ (different label) get DISTINCT entries.

    The trace digest is identical; only ``bundle.label`` differs, which changes the
    rendered prompt. The verdict for one bundle must never overwrite or be served for the
    other -- a substring/digest-only key would collide here and corrupt a bisect.
    """
    cache_path = tmp_path / "judge.json"
    t = _trace("same-output")

    # Bundle A renders to a prompt whose verdict is BAD.
    backend_a = FakeLLM(lambda s, u: "BAD", configured_model="m")
    judge_a = LLMJudge(backend_a, cache=JudgeCache(cache_path))
    assert judge_a.judge(t, _bundle(final="same-output", label="bundle-A")) is Verdict.BAD

    # Bundle B (identical trace, DIFFERENT label) renders to a different prompt; its
    # verdict is GOOD. It must be judged afresh, not served Bundle A's cached BAD.
    b_calls = {"n": 0}

    def b_responder(system: str, user: str) -> str:
        b_calls["n"] += 1
        return "GOOD"

    backend_b = FakeLLM(b_responder, configured_model="m")
    judge_b = LLMJudge(backend_b, cache=JudgeCache(cache_path))
    assert judge_b.judge(t, _bundle(final="same-output", label="bundle-B")) is Verdict.GOOD
    assert b_calls["n"] == 1  # NOT served from A's entry -> the backend was consulted

    # Two distinct entries coexist; neither overwrote the other.
    assert len(JudgeCache(cache_path)) == 2
    # And A's verdict is unchanged after B was judged (no overwrite).
    assert judge_a.judge(t, _bundle(final="same-output", label="bundle-A")) is Verdict.BAD


def test_same_input_hits_single_entry_after_resolution(tmp_path: Path) -> None:
    """After the model id resolves, repeated judges of the same input hit ONE entry.

    A second judge configured with the resolved id and the same rendered prompt must read
    the cached verdict without re-calling the backend, leaving exactly one entry.
    """
    cache_path = tmp_path / "judge.json"
    t = _trace("answer")
    bundle = _bundle(label="case")

    first_calls = {"n": 0}

    def first_responder(system: str, user: str) -> str:
        first_calls["n"] += 1
        return "BAD"

    backend = FakeLLM(
        first_responder,
        configured_model="judge-latest",
        resolved_model="judge-2026-06-01",
    )
    judge = LLMJudge(backend, cache=JudgeCache(cache_path))
    assert judge.judge(t, bundle) is Verdict.BAD
    assert first_calls["n"] == 1
    # The same judge re-judging the same input is served from cache (no second call).
    assert judge.judge(t, bundle) is Verdict.BAD
    assert first_calls["n"] == 1
    assert len(JudgeCache(cache_path)) == 1


def test_judge_version_bump_invalidates_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "judge.json"
    backend = FakeLLM(lambda s, u: "BAD", configured_model="m")
    j1 = LLMJudge(backend, judge_version=1, cache=JudgeCache(cache_path))
    t = _trace("answer")
    assert j1.judge(t, _bundle()) is Verdict.BAD

    # A new judge version uses a different key -> separate entry (re-judged).
    calls = {"n": 0}

    def responder(system: str, user: str) -> str:
        calls["n"] += 1
        return "GOOD"

    j2 = LLMJudge(
        FakeLLM(responder, configured_model="m"),
        judge_version=2,
        cache=JudgeCache(cache_path),
    )
    assert j2.judge(t, _bundle()) is Verdict.GOOD
    assert calls["n"] == 1
    assert len(JudgeCache(cache_path)) == 2


def test_judge_cache_persists_to_disk(tmp_path: Path) -> None:
    cache_path = tmp_path / "sub" / "judge.json"
    backend = FakeLLM(lambda s, u: "BAD", configured_model="m")
    judge = LLMJudge(backend, cache=JudgeCache(cache_path))
    judge.judge(_trace("z"), _bundle())
    assert cache_path.exists()
    # A fresh cache loaded from disk sees the entry.
    assert len(JudgeCache(cache_path)) == 1


def test_in_memory_cache_has_no_path() -> None:
    cache = JudgeCache()
    cache.set("k", Verdict.GOOD)
    assert cache.get("k") is Verdict.GOOD
    assert cache.path is None


def test_set_alias_is_idempotent_and_does_not_rewrite(tmp_path: Path) -> None:
    """Recording the same alias->resolved mapping twice is a no-op (no redundant disk write)."""
    cache_path = tmp_path / "judge.json"
    cache = JudgeCache(cache_path)
    cache.set_alias("judge-latest", "judge-2026-06-01")
    assert cache.resolved_alias("judge-latest") == "judge-2026-06-01"

    first_mtime = cache_path.stat().st_mtime_ns
    # Re-recording the identical mapping must early-return without rewriting the file.
    cache.set_alias("judge-latest", "judge-2026-06-01")
    assert cache_path.stat().st_mtime_ns == first_mtime
    assert cache.resolved_alias("judge-latest") == "judge-2026-06-01"
    # An unknown alias resolves to None.
    assert cache.resolved_alias("other-latest") is None

    # A fresh cache loaded from disk still sees the persisted alias mapping.
    assert JudgeCache(cache_path).resolved_alias("judge-latest") == "judge-2026-06-01"


def test_parse_verdict_empty_string_is_skip() -> None:
    """An empty completion is undecidable -> SKIP (guards the bare-token short-circuit)."""
    judge = LLMJudge(FakeLLM(lambda s, u: ""), cache=JudgeCache())
    assert judge.judge(_trace(), _bundle()) is Verdict.SKIP
