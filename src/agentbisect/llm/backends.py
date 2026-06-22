"""LLM backend implementations behind the :class:`LLMBackend` protocol.

Every backend reports a ``resolved_model_id`` after its first completion: the concrete
model string the provider actually used (e.g. ``gpt-4o-2024-08-06``), never a moving
alias (``gpt-4o``/``*-latest``). The judge folds this into its cache key so the same
candidate is never re-judged differently across machines or CI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AnthropicBackend",
    "FakeLLM",
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "Responder",
]

#: A FakeLLM responder maps (system, user) -> reply text.
Responder = Callable[[str, str], str]


@runtime_checkable
class LLMBackend(Protocol):
    """A minimal chat backend for the judge.

    ``complete`` runs a single deterministic completion (the judge always uses
    ``temperature=0``). ``resolved_model_id`` returns the concrete model id once known
    (after the first call); before that it may return the configured alias.
    """

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system+user prompt."""
        ...

    @property
    def resolved_model_id(self) -> str:
        """The concrete model id the backend used (post first call)."""
        ...


class FakeLLM:
    """Deterministic, offline backend for tests and demos.

    ``responder`` maps ``(system, user) -> reply``. ``configured_model`` is what the
    user asked for (possibly an alias); ``resolved_model`` is what the backend "reports"
    after its first call -- letting tests prove that an alias and its resolved id collapse
    to a single judge cache entry.
    """

    def __init__(
        self,
        responder: Responder,
        *,
        configured_model: str = "fake-model",
        resolved_model: str | None = None,
    ) -> None:
        self._responder = responder
        self._configured = configured_model
        self._resolved = resolved_model or configured_model
        self._called = False

    def complete(self, system: str, user: str) -> str:
        self._called = True
        return self._responder(system, user)

    @property
    def resolved_model_id(self) -> str:
        # Report the alias until the first call resolves the concrete id.
        return self._resolved if self._called else self._configured


class OpenAIBackend:  # pragma: no cover - exercised only with network/credentials
    """OpenAI Chat Completions backend (requires the ``[openai]`` extra)."""

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        self._model = model
        self._resolved: str | None = None
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client: Any = client

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._resolved = getattr(resp, "model", None) or self._model
        content = resp.choices[0].message.content
        return content or ""

    @property
    def resolved_model_id(self) -> str:
        return self._resolved or self._model


class AnthropicBackend:  # pragma: no cover - exercised only with network/credentials
    """Anthropic Messages backend (requires the ``[anthropic]`` extra)."""

    def __init__(self, model: str, *, client: Any | None = None, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._resolved: str | None = None
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self._client: Any = client

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._resolved = getattr(resp, "model", None) or self._model
        parts = [getattr(block, "text", "") for block in resp.content]
        return "".join(parts)

    @property
    def resolved_model_id(self) -> str:
        return self._resolved or self._model


class OllamaBackend:  # pragma: no cover - exercised only with a local Ollama daemon
    """Local Ollama backend via its HTTP API (no extra dependency)."""

    def __init__(self, model: str, *, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._resolved: str | None = None

    def complete(self, system: str, user: str) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self._model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self._resolved = data.get("model", self._model)
        return str(data.get("response", ""))

    @property
    def resolved_model_id(self) -> str:
        return self._resolved or self._model
