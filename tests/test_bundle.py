"""Tests for RunBundle (de)serialization, integrity, versioning, and fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbisect.bundle import (
    BUNDLE_FILENAME,
    BundleIntegrityError,
    BundleVersionError,
    load_bundle,
    make_bundle,
    save_bundle,
)
from agentbisect.types import AgentConfig, LlmStep, ToolSchema, ToolStep, Trace


def _sample_trace() -> Trace:
    return Trace(
        steps=(
            LlmStep(index=0, role="assistant", content="thinking"),
            ToolStep(index=1, tool="search", args={"q": "refund"}, output="policy text", ok=True),
            LlmStep(index=2, role="assistant", content="done"),
        ),
        final_output="the refund policy is 30 days",
    )


def _sample_config() -> AgentConfig:
    return AgentConfig(
        system_prompt="You are a support agent. Always state the refund policy.",
        model="gpt-4o-2024-08-06",
        params={"temperature": 0, "max_tokens": 256},
        tool_schemas=(ToolSchema(name="search", schema={"type": "object"}, version="3"),),
        retrieval_ref="snap-2026-06",
    )


def test_roundtrip_preserves_everything(tmp_path: Path) -> None:
    bundle = make_bundle(config=_sample_config(), trace=_sample_trace(), label="case-1")
    out = save_bundle(bundle, tmp_path / "bundle")
    loaded = load_bundle(out)

    assert loaded.config == bundle.config
    assert loaded.trace == bundle.trace
    assert loaded.label == "case-1"
    assert loaded.integrity == bundle.integrity
    # The tool-schema alias survives the on-disk 'schema' name.
    assert loaded.config.tool_schemas[0].json_schema == {"type": "object"}


def test_on_disk_uses_schema_alias(tmp_path: Path) -> None:
    bundle = make_bundle(config=_sample_config(), trace=_sample_trace(), label="x")
    out = save_bundle(bundle, tmp_path / "b")
    raw = json.loads((out / BUNDLE_FILENAME).read_text())
    assert raw["config"]["tool_schemas"][0]["schema"] == {"type": "object"}
    assert "json_schema" not in raw["config"]["tool_schemas"][0]


def test_load_accepts_direct_json_path(tmp_path: Path) -> None:
    bundle = make_bundle(config=_sample_config(), trace=_sample_trace(), label="x")
    out = save_bundle(bundle, tmp_path / "b")
    loaded = load_bundle(out / BUNDLE_FILENAME)
    assert loaded.label == "x"


def test_integrity_detects_tampering(tmp_path: Path) -> None:
    bundle = make_bundle(config=_sample_config(), trace=_sample_trace(), label="x")
    out = save_bundle(bundle, tmp_path / "b")
    file_path = out / BUNDLE_FILENAME
    data = json.loads(file_path.read_text())
    data["trace"]["final_output"] = "TAMPERED"  # integrity stays stale
    file_path.write_text(json.dumps(data))

    with pytest.raises(BundleIntegrityError):
        load_bundle(out)


def test_future_version_raises(tmp_path: Path) -> None:
    bundle = make_bundle(config=_sample_config(), trace=_sample_trace(), label="x")
    out = save_bundle(bundle, tmp_path / "b")
    file_path = out / BUNDLE_FILENAME
    data = json.loads(file_path.read_text())
    data["version"] = 999
    file_path.write_text(json.dumps(data))

    with pytest.raises(BundleVersionError):
        load_bundle(out)


def test_fingerprint_is_deterministic_and_order_independent() -> None:
    a = AgentConfig(
        system_prompt="p",
        model="m",
        params={"temperature": 0, "top_p": 0.9},
    )
    b = AgentConfig(
        system_prompt="p",
        model="m",
        params={"top_p": 0.9, "temperature": 0},  # different insertion order
    )
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_prompt() -> None:
    a = AgentConfig(system_prompt="p1", model="m")
    b = AgentConfig(system_prompt="p2", model="m")
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_float_canonicalization() -> None:
    # Floats equal to 9 significant digits collapse; meaningfully different ones do not.
    a = AgentConfig(system_prompt="p", model="m", params={"temperature": 0.1})
    b = AgentConfig(system_prompt="p", model="m", params={"temperature": 0.1000000000001})
    c = AgentConfig(system_prompt="p", model="m", params={"temperature": 0.2})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_trace_digest_is_stable() -> None:
    t1 = _sample_trace()
    t2 = _sample_trace()
    assert t1.digest() == t2.digest()


def test_tool_schema_property_alias() -> None:
    ts = ToolSchema(name="s", schema={"k": "v"})
    assert ts.schema == {"k": "v"}
    assert ts.schema == ts.json_schema


def test_load_corrupt_json_raises_bundle_error(tmp_path: Path) -> None:
    from agentbisect.bundle import BundleError

    out = tmp_path / "b"
    out.mkdir()
    (out / BUNDLE_FILENAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(BundleError):
        load_bundle(out)
