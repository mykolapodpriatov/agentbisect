"""Oracles that judge a replayed candidate good / bad / skip.

Implementations:

* :class:`AssertionOracle` -- a user predicate over the trace / final output.
* :class:`FakeOracle` -- scripted verdicts for tests (by ref or in call order).
* :class:`LLMJudge` -- a pinned judge prompt + model deciding good/bad, with a
  deterministic, CI-safe cache.

LLMJudge cache key
------------------
``sha256(full_rendered_prompt + resolved_model_id + judge_version)``

The ``full_rendered_prompt`` is the exact text sent to the model -- the judge
instruction plus the rendered user payload, which already incorporates the trace
content, the bundle ``label``, and the judge instructions. Two bundles whose prompts
render differently (e.g. a different ``label``) therefore land on *distinct* keys and
can never collide or overwrite each other's verdict.

The ``resolved_model_id`` is the concrete model string the backend reports *after its
first call* (not a ``*-latest`` alias). It is resolved lazily and memoized on the judge
instance, so the cache key never keys on a moving alias: configuring the judge with an
alias and a backend that resolves to a concrete id produces a *single* cache entry --
the resolved id -- so a candidate is never double-judged.

So that the collapse also holds for a *cold* judge (a fresh instance/process), the cache file
persists an alias->resolved-id map alongside the verdicts. The first time any judge resolves
an alias it records the mapping; thereafter a cold instance configured with that alias reads
the persisted resolved id, builds the resolved-id key, and hits the existing verdict entry
with *no* extra backend call. Verdicts persist to ``.agentbisect_cache/judge.json``
(committable) for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from .llm import LLMBackend
from .types import RunBundle, Trace, Verdict

__all__ = [
    "DEFAULT_JUDGE_PROMPT",
    "AssertionOracle",
    "FakeOracle",
    "JudgeCache",
    "LLMJudge",
    "Oracle",
]

#: Default judge instruction. Bumping ``judge_version`` invalidates cached verdicts.
DEFAULT_JUDGE_PROMPT = (
    "You are a strict regression judge for an AI agent. Given the agent's final output "
    "and trace, decide whether the run is GOOD (correct/acceptable) or BAD (regressed). "
    "If you cannot decide confidently, answer SKIP. Answer with exactly one word: "
    "GOOD, BAD, or SKIP."
)


@runtime_checkable
class Oracle(Protocol):
    """Judges a candidate's replayed trace against the captured bundle."""

    def judge(self, trace: Trace, bundle: RunBundle) -> Verdict:
        """Return ``good``/``bad``/``skip`` for ``trace`` (context: ``bundle``)."""
        ...


class AssertionOracle:
    """Judges via a user predicate over the trace and final output.

    ``predicate`` returns ``True`` for a good run and ``False`` for a bad one. To express
    "cannot decide", raise :class:`Undecidable` (mapped to ``skip``).
    """

    class Undecidable(Exception):
        """Raise inside a predicate to yield a ``skip`` verdict."""

    def __init__(self, predicate: Callable[[Trace, RunBundle], bool]) -> None:
        self._predicate = predicate

    def judge(self, trace: Trace, bundle: RunBundle) -> Verdict:
        try:
            ok = self._predicate(trace, bundle)
        except AssertionOracle.Undecidable:
            return Verdict.SKIP
        return Verdict.GOOD if ok else Verdict.BAD


class FakeOracle:
    """Scripted oracle for tests.

    Provide ``by_ref`` (config fingerprint or candidate ref -> verdict) and/or
    ``sequence`` (verdicts returned in call order). A ``by_ref`` value may be a list to
    model a *flaky* candidate that alternates verdicts across repeated probes. The trace's
    ``final_output`` is also accepted as a key, which is how the FakeAgent-driven demo maps
    behavior to verdicts.
    """

    def __init__(
        self,
        *,
        by_output: dict[str, Verdict] | None = None,
        by_ref: dict[str, Verdict | list[Verdict]] | None = None,
        sequence: list[Verdict] | None = None,
        default: Verdict = Verdict.SKIP,
    ) -> None:
        self._by_output = by_output or {}
        self._by_ref = by_ref or {}
        self._sequence = list(sequence or [])
        self._default = default
        self._seq_pos = 0
        self._flaky_pos: dict[str, int] = {}

    def judge(self, trace: Trace, bundle: RunBundle) -> Verdict:
        if self._sequence:
            verdict = self._sequence[min(self._seq_pos, len(self._sequence) - 1)]
            self._seq_pos += 1
            return verdict
        fp = bundle.config.fingerprint()
        if fp in self._by_ref:
            return self._resolve_ref(fp)
        if trace.final_output in self._by_output:
            return self._by_output[trace.final_output]
        return self._default

    def _resolve_ref(self, key: str) -> Verdict:
        value = self._by_ref[key]
        if isinstance(value, list):
            pos = self._flaky_pos.get(key, 0)
            self._flaky_pos[key] = pos + 1
            return value[pos % len(value)]
        return value


#: Top-level keys under which the committable cache file nests its two sections. They are
#: reserved (never collide with a 64-hex sha256 verdict key) and absent from the legacy flat
#: ``{key: verdict}`` layout, which is still read transparently.
_VERDICTS_SECTION = "verdicts"
_ALIASES_SECTION = "model_aliases"


class JudgeCache:
    """A committable JSON cache of judge verdicts plus an alias->resolved-model-id map.

    Two sections persist side by side in the file:

    * ``verdicts``: ``cache_key -> verdict`` (keyed by the resolved-model-id cache key).
    * ``model_aliases``: ``configured_alias -> resolved_model_id``, recorded the first time
      any judge resolves an alias. A cold judge configured with that alias can then build the
      resolved-id key *without* a backend call, so it hits an existing verdict entry instead of
      re-querying (see :meth:`LLMJudge.judge`).

    The legacy flat ``{cache_key: verdict}`` layout (no sections) is still read transparently.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._data: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        if self.path is not None and self.path.exists():
            self._load(json.loads(self.path.read_text(encoding="utf-8")))

    def _load(self, raw: dict[str, object]) -> None:
        """Populate from disk, accepting both the sectioned and the legacy flat layout."""
        verdicts = raw.get(_VERDICTS_SECTION)
        aliases = raw.get(_ALIASES_SECTION)
        if isinstance(verdicts, dict) or isinstance(aliases, dict):
            if isinstance(verdicts, dict):
                self._data = {str(k): str(v) for k, v in verdicts.items()}
            if isinstance(aliases, dict):
                self._aliases = {str(k): str(v) for k, v in aliases.items()}
        else:  # legacy: the whole file is the flat verdict map.
            self._data = {str(k): str(v) for k, v in raw.items()}

    def get(self, key: str) -> Verdict | None:
        raw = self._data.get(key)
        return Verdict(raw) if raw is not None else None

    def set(self, key: str, verdict: Verdict) -> None:
        self._data[key] = verdict.value
        self._flush()

    def resolved_alias(self, alias: str) -> str | None:
        """Return the persisted resolved model id for ``alias``, or ``None`` if unrecorded."""
        return self._aliases.get(alias)

    def set_alias(self, alias: str, resolved_model_id: str) -> None:
        """Persist (idempotently) that ``alias`` resolves to ``resolved_model_id``."""
        if self._aliases.get(alias) == resolved_model_id:
            return
        self._aliases[alias] = resolved_model_id
        self._flush()

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {_VERDICTS_SECTION: self._data, _ALIASES_SECTION: self._aliases}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._data)


#: The judge is instructed to answer with exactly one of these words.
_VERDICT_WORDS = {
    "good": Verdict.GOOD,
    "bad": Verdict.BAD,
    "skip": Verdict.SKIP,
}

#: Punctuation/markdown a real model commonly wraps a one-word verdict in, e.g. ``GOOD.``,
#: ``**good**``, ``` `bad` ```, ``"skip"``. Stripped (one layer) from both ends of a single
#: token before matching. NOT used for substring matching -- only single-token unwrapping.
_VERDICT_WRAPPERS = ".,!?:;\"'`*"


def _parse_verdict(text: str) -> Verdict:
    """Map a judge's answer to a Verdict, requiring an exact one-word verdict.

    Real models routinely wrap the one-word verdict in punctuation or markdown -- ``GOOD.``,
    ``**good**``, ``` `bad` ```, ``"skip"`` -- so a single token is normalized before the
    lookup: surrounding whitespace is stripped, then one wrapping layer of common
    punctuation/markdown (``. , ! ? : ; " '`` backtick ``*``) is stripped from both ends, then
    it is lowercased. Only an exact single verdict word matches.

    Anything that is not a single bare token -- a multi-word phrase that merely *contains* a
    verdict word, such as ``"not bad"`` or ``"good enough"``, or a glued-together token like
    ``"goodbad"`` -- maps to ``skip`` rather than risking a confidently-wrong substring match.
    """
    token = text.strip()
    # Multi-word answers are never a verdict: reject before unwrapping so that "not bad" and
    # "good enough" stay SKIP and are never substring-matched.
    if not token or any(ch.isspace() for ch in token):
        return Verdict.SKIP
    token = token.strip(_VERDICT_WRAPPERS).lower()
    return _VERDICT_WORDS.get(token, Verdict.SKIP)


class LLMJudge:
    """An LLM-backed oracle with a deterministic, resolved-model-id-keyed cache.

    The cache key is computed from ``resolved_model_id``, so an alias and the concrete id it
    resolves to share one entry and a candidate is judged once. The cache file also persists an
    alias->resolved-id map, so this collapse holds even across cold instances/processes: a fresh
    judge configured with a moving alias reads the persisted resolved id, builds the resolved-id
    key, and hits the existing entry with no extra backend call (the alias is resolved by a
    single bootstrapping call only the first time it is seen anywhere).
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        judge_version: int = 1,
        cache: JudgeCache | None = None,
    ) -> None:
        self._backend = backend
        self._judge_prompt = judge_prompt
        self._judge_version = judge_version
        self._cache = cache if cache is not None else JudgeCache()
        #: Lazily resolved + memoized concrete model id (set after the first backend call).
        self._resolved_model_id: str | None = None

    def _model_id(self) -> str:
        """Return the best-known model id for *looking up* a cached verdict.

        Resolution order: the id memoized on this instance (post first call); else the
        resolved id persisted for the configured alias in the cache file -- this lets a cold
        instance configured with a moving alias build the resolved-id key with NO backend
        call and hit an existing entry; else whatever the backend reports (the configured
        id/alias), used only to look up a possibly-existing entry.
        """
        if self._resolved_model_id is not None:
            return self._resolved_model_id
        configured = self._backend.resolved_model_id
        persisted = self._cache.resolved_alias(configured)
        if persisted is not None:
            # Memoize so the post-call re-key (and any later miss path) stays consistent.
            self._resolved_model_id = persisted
            return persisted
        return configured

    def _cache_key(self, rendered_prompt: str, model_id: str) -> str:
        """Key a verdict on the exact prompt text, the resolved model id, and the version.

        Including the full rendered prompt means two bundles whose prompts differ (e.g. a
        different ``label``) never share a key; identical inputs collapse to one entry.
        """
        material = "␟".join([rendered_prompt, model_id, str(self._judge_version)])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _render_user(self, trace: Trace, bundle: RunBundle) -> str:
        return json.dumps(
            {
                "task_label": bundle.label,
                "final_output": trace.final_output,
                "steps": [s.model_dump(mode="json") for s in trace.steps],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _rendered_prompt(self, user: str) -> str:
        """The full text sent to the model: the judge instruction plus the user payload."""
        return f"{self._judge_prompt}␞{user}"

    def judge(self, trace: Trace, bundle: RunBundle) -> Verdict:
        """Return a cached or freshly judged verdict for ``trace``.

        The key is built from the full rendered prompt (instruction + user payload, which
        already carries the trace content + ``bundle.label``) and the resolved model id. The
        lookup id comes from :meth:`_model_id`, which consults the persisted alias->resolved-id
        map so that even a cold instance configured with a moving alias keys on the resolved id
        and hits an existing entry with no backend call.

        On a genuine miss the backend is called once (resolving and memoizing the concrete id);
        the verdict is stored under the resolved-id key and the alias->resolved mapping is
        persisted, so future cold instances configured with the alias skip the backend entirely.
        """
        # Capture the configured id BEFORE any call: pre-resolution the backend reports the
        # alias, which is the key under which we persist the alias->resolved mapping.
        configured_id = self._backend.resolved_model_id

        user = self._render_user(trace, bundle)
        rendered = self._rendered_prompt(user)

        pre_key = self._cache_key(rendered, self._model_id())
        cached = self._cache.get(pre_key)
        if cached is not None:
            return cached

        answer = self._backend.complete(self._judge_prompt, user)
        verdict = _parse_verdict(answer)

        # The backend has now resolved the concrete model id; memoize it and re-key so an
        # alias and the id it resolves to collapse to a single cache entry.
        self._resolved_model_id = self._backend.resolved_model_id
        resolved_key = self._cache_key(rendered, self._resolved_model_id)
        self._cache.set(resolved_key, verdict)
        # Persist the alias->resolved mapping so a future COLD instance configured with the
        # alias can build the resolved-id key and hit this entry without a backend call.
        if configured_id != self._resolved_model_id:
            self._cache.set_alias(configured_id, self._resolved_model_id)
        return verdict
