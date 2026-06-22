"""Tests for the canonicalization helpers (determinism discipline)."""

from __future__ import annotations

from agentbisect.canonical import canonical_float, canonical_json, stable_hash


def test_canonical_float_nine_sig_figs() -> None:
    assert canonical_float(0.1) == canonical_float(0.1000000000001)
    assert canonical_float(0.1) != canonical_float(0.2)


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_handles_sets_order_independent() -> None:
    # Sets are canonicalized by sorting their elements' canonical form.
    assert canonical_json({1, 2, 3}) == canonical_json({3, 2, 1})


def test_canonical_json_nested_floats() -> None:
    a = canonical_json({"params": {"t": 0.1}})
    b = canonical_json({"params": {"t": 0.1000000000001}})
    assert a == b


def test_canonical_json_preserves_bool() -> None:
    out = canonical_json({"flag": True, "off": False})
    assert "true" in out
    assert "false" in out


def test_canonical_json_tuple_like_list() -> None:
    assert canonical_json((1, 2, 3)) == canonical_json([1, 2, 3])


def test_stable_hash_is_hex_sha256() -> None:
    h = stable_hash({"a": 1})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})
