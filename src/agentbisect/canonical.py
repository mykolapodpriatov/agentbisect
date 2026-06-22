"""Canonicalization helpers shared by fingerprints, hashes, and tool-mock matching.

Determinism discipline (used everywhere a stable hash or key is needed):

* mappings are serialized with sorted keys,
* floats are formatted with ``{:.9g}`` so that ``1.0`` and ``1.00000000001`` are
  distinguished but trivial float noise does not change a key,
* the result is compact (no insignificant whitespace) and ``ensure_ascii`` so the
  byte representation is identical across platforms and locales.

The exact same routine is used by :class:`~agentbisect.types.AgentConfig.fingerprint`,
the :class:`~agentbisect.types.RunBundle` integrity hash, the tool-mock matcher in
:mod:`agentbisect.mock_tools`, and the judge cache key in :mod:`agentbisect.oracle`.
Keeping a single implementation guarantees the same value is computed in capture, in
replay, and in CI.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_float", "canonical_json", "stable_hash"]


def canonical_float(value: float) -> str:
    """Format a float deterministically with 9 significant digits."""
    return f"{value:.9g}"


def _canonicalize(value: Any) -> Any:
    """Recursively convert a value into a JSON-canonical, hash-stable shape.

    Floats become their ``.9g`` string form, mappings get sorted keys, and sets are
    sorted by their canonical form so that element order never affects the result.
    """
    if isinstance(value, float):
        return canonical_float(value)
    if isinstance(value, bool):
        # bool is a subclass of int; keep it as-is (JSON true/false).
        return value
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonicalize(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True),
        )
    return value


def canonical_json(value: Any) -> str:
    """Return a deterministic, compact JSON string for ``value``.

    Two values that are semantically equal under the determinism discipline produce
    byte-identical output regardless of dict insertion order or float noise below the
    9-significant-digit threshold.
    """
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_hash(value: Any) -> str:
    """Return a hex SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
