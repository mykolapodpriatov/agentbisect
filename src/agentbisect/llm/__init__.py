"""LLM backends for the :class:`~agentbisect.oracle.LLMJudge`.

A backend exposes the judge model behind a tiny :class:`LLMBackend` protocol and,
crucially, reports the *resolved* concrete model id (not a ``*-latest`` alias) after its
first call, so the judge cache key is stable across machines and CI.

Backends:

* :class:`FakeLLM` -- deterministic, offline, scripted; used in all tests.
* :class:`OpenAIBackend` -- OpenAI Chat Completions (optional ``[openai]`` extra).
* :class:`AnthropicBackend` -- Anthropic Messages (optional ``[anthropic]`` extra).
* :class:`OllamaBackend` -- local Ollama via its HTTP API (no extra needed).
"""

from __future__ import annotations

from .backends import (
    AnthropicBackend,
    FakeLLM,
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
)

__all__ = [
    "AnthropicBackend",
    "FakeLLM",
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
]
