"""Deterministic, network-free agent for tests and demos.

The :class:`FakeAgent` is a *pure function* of ``(config, tool_outputs)``: given the
same config and the same tool outputs it always produces the same trace. This is what
makes the whole capture -> replay -> bisect pipeline reproducible offline.

Behavior is driven by a tiny embedded "program" carried in
``config.params["program"]`` (a list of step specs). This lets a test or demo model a
regression purely through config: e.g. a prompt commit that drops a policy clause makes
the final answer change, and the FakeAgent reflects that deterministically.

A program is a list of dicts, each either::

    {"tool": "search", "args": {...}}        # a tool call (output comes from the provider)
    {"llm": "assistant message text"}        # an llm turn (literal content)

The final output is ``config.params["final"]`` if present, otherwise a deterministic
string derived from the system prompt and the tool outputs seen. When ``final`` (or any
literal content) contains the marker ``{prompt_has:TOKEN}`` it is replaced by ``yes``/``no``
depending on whether ``TOKEN`` appears in the system prompt -- a compact way to encode a
prompt-driven regression.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import AgentConfig, LlmStep, Step, ToolStep, Trace
from .tools import ToolProvider

__all__ = ["FakeAgent"]

_MARKER = re.compile(r"\{prompt_has:([^}]+)\}")


class FakeAgent:
    """A deterministic agent driven by a program embedded in ``config.params``."""

    def _render(self, text: str, system_prompt: str) -> str:
        """Resolve ``{prompt_has:TOKEN}`` markers against the system prompt."""

        def repl(match: re.Match[str]) -> str:
            token = match.group(1)
            return "yes" if token in system_prompt else "no"

        return _MARKER.sub(repl, text)

    def run(self, config: AgentConfig, tools: ToolProvider) -> Trace:
        """Execute the embedded program, recording each step with a stable index."""
        program: list[dict[str, Any]] = list(config.params.get("program", []))
        steps: list[Step] = []
        seen: dict[tuple[str, str], int] = {}
        tool_results: list[str] = []

        for idx, spec in enumerate(program):
            if "tool" in spec:
                tool = str(spec["tool"])
                args = dict(spec.get("args", {}))
                output = tools.call(tool, args)
                # occurrence = how many prior steps invoked the same (tool, canon args).
                from ..canonical import canonical_json

                key = (tool, canonical_json(args))
                occurrence = seen.get(key, 0)
                seen[key] = occurrence + 1
                steps.append(
                    ToolStep(
                        index=idx,
                        tool=tool,
                        args=args,
                        output=output,
                        ok=True,
                        occurrence=occurrence,
                    )
                )
                tool_results.append(str(output))
            elif "llm" in spec:
                content = self._render(str(spec["llm"]), config.system_prompt)
                steps.append(LlmStep(index=idx, role="assistant", content=content))
            else:  # pragma: no cover - defensive; programs are author-controlled
                raise ValueError(f"Invalid program step at {idx}: {spec!r}")

        if "final" in config.params:
            final = self._render(str(config.params["final"]), config.system_prompt)
        else:
            final = "|".join(tool_results)
        return Trace(steps=tuple(steps), final_output=final)
