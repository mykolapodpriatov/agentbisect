"""Smoke test for the offline example demo (keeps it runnable in CI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "refund_regression.py"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example file not present")
def test_example_runs_and_finds_first_bad(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("refund_regression", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()
    out = capsys.readouterr().out
    assert "captured baseline: refund=yes" in out
    assert "first bad change: rev-2" in out
    assert "last good: rev-1" in out
    assert "refund=yes' -> 'refund=no'" in out
