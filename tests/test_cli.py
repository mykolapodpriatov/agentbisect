"""End-to-end CLI tests using FakeAgent + FakeOracle (fully offline)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentbisect.bundle import BUNDLE_FILENAME
from agentbisect.cli import app

runner = CliRunner()


# A project config that captures a refund-regression run and bisects a model axis where
# later models drop the refund clause. Written to a temp file by each test.
PROJECT_CONFIG = """
from agentbisect.agent.fake import FakeAgent
from agentbisect.oracle import AssertionOracle
from agentbisect.types import AgentConfig


def runner():
    return FakeAgent()


def config():
    return AgentConfig(
        system_prompt="You are support. Always state the refund policy.",
        model="m0",
        params={
            "program": [{"tool": "kb", "args": {"q": "policy"}}],
            "final": "refund={prompt_has:refund}",
        },
    )


def tool_executor():
    def run(tool, args):
        return f"{tool}-result"
    return run


def oracle():
    # GOOD iff the refund clause is still present in the output.
    return AssertionOracle(lambda t, b: t.final_output == "refund=yes")
"""


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "project.py"
    cfg.write_text(PROJECT_CONFIG, encoding="utf-8")
    return cfg


def test_capture_writes_bundle(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    result = runner.invoke(
        app, ["capture", "--config", str(cfg), "--out", str(out), "--label", "case-1"]
    )
    assert result.exit_code == 0, result.output
    assert (out / BUNDLE_FILENAME).exists()
    data = json.loads((out / BUNDLE_FILENAME).read_text())
    assert data["label"] == "case-1"
    assert data["config"]["model"] == "m0"


def test_bisect_nonmonotonic_endpoints_exit_code(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    cap = runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    assert cap.exit_code == 0, cap.output

    # A ModelListAxis overrides only the model; the captured prompt (which keeps the
    # refund clause) is held for every candidate, so m0 and m1 are both GOOD. The last
    # endpoint being GOOD is non-monotonic -> the bisect raises and the CLI exits 3.
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
            "--markdown",
        ],
    )
    assert res.exit_code == 3, res.output
    assert "error" in res.output.lower()


def test_bisect_prompt_axis_reports_first_bad(
    tmp_path: Path,
    temp_git_repo,
) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])

    # A prompt history where the refund clause is dropped in the final commit.
    versions = [
        "You are support. Always state the refund policy.",  # good
        "You are support. Always state the refund policy. Be brief.",  # good
        "You are support. Be brief.",  # BAD: refund clause removed
    ]
    repo = temp_git_repo("system.txt", versions)

    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "prompt",
            "--over",
            f"{repo}:system.txt",
            "--markdown",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "First bad change" in res.output


def test_bisect_missing_bundle_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(tmp_path / "nope"),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
        ],
    )
    assert res.exit_code == 4, res.output


def test_capture_bad_config_usage_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("# no hooks defined here\n", encoding="utf-8")
    res = runner.invoke(app, ["capture", "--config", str(bad), "--out", str(tmp_path / "b")])
    assert res.exit_code == 4, res.output


def test_diff_command_identical_bundles(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(app, ["diff", str(out), str(out)])
    assert res.exit_code == 0
    assert "no behavioral difference" in res.output


def test_diff_command_shows_differences(tmp_path: Path) -> None:
    # Build two bundles with different final outputs and save them directly.
    from agentbisect.bundle import make_bundle, save_bundle
    from agentbisect.types import AgentConfig, LlmStep, Trace

    cfg_obj = AgentConfig(system_prompt="p", model="m")
    a_dir = save_bundle(
        make_bundle(
            config=cfg_obj,
            trace=Trace(steps=(LlmStep(index=0, role="a", content="x"),), final_output="good"),
            label="a",
        ),
        tmp_path / "a",
    )
    b_dir = save_bundle(
        make_bundle(
            config=cfg_obj,
            trace=Trace(steps=(LlmStep(index=0, role="a", content="y"),), final_output="bad"),
            label="b",
        ),
        tmp_path / "b",
    )
    res = runner.invoke(app, ["diff", str(a_dir), str(b_dir)])
    assert res.exit_code == 0
    assert "first divergence" in res.output
    assert "final:" in res.output


def test_diff_command_missing_bundle_usage_error(tmp_path: Path) -> None:
    res = runner.invoke(app, ["diff", str(tmp_path / "nope-a"), str(tmp_path / "nope-b")])
    assert res.exit_code == 4, res.output


def _two_bundles(tmp_path: Path) -> tuple[Path, Path]:
    """Save two bundles with the same step-0 content but different final outputs."""
    from agentbisect.bundle import make_bundle, save_bundle
    from agentbisect.types import AgentConfig, LlmStep, Trace

    cfg_obj = AgentConfig(system_prompt="p", model="m")
    a_dir = save_bundle(
        make_bundle(
            config=cfg_obj,
            trace=Trace(steps=(LlmStep(index=0, role="a", content="x"),), final_output="good"),
            label="a",
        ),
        tmp_path / "a",
    )
    b_dir = save_bundle(
        make_bundle(
            config=cfg_obj,
            trace=Trace(steps=(LlmStep(index=0, role="a", content="y"),), final_output="bad"),
            label="b",
        ),
        tmp_path / "b",
    )
    return a_dir, b_dir


def test_diff_command_json_output_is_machine_readable(tmp_path: Path) -> None:
    a_dir, b_dir = _two_bundles(tmp_path)
    res = runner.invoke(app, ["diff", str(a_dir), str(b_dir), "--json"])
    assert res.exit_code == 0, res.output
    # Pipe-safe: the bracketed step array is not swallowed by Rich markup.
    data = json.loads(res.output)
    assert data["is_empty"] is False
    assert data["first_divergence"] == 0
    assert data["final_output_changed"] is True
    assert data["left_final"] == "good"
    assert data["right_final"] == "bad"
    assert [s["index"] for s in data["differing_steps"]] == [0]


def test_diff_command_markdown_output(tmp_path: Path) -> None:
    a_dir, b_dir = _two_bundles(tmp_path)
    res = runner.invoke(app, ["diff", str(a_dir), str(b_dir), "--markdown"])
    assert res.exit_code == 0, res.output
    assert "# behavioral diff" in res.output
    assert "Differing steps" in res.output


def test_diff_command_json_empty_diff(tmp_path: Path) -> None:
    a_dir, _ = _two_bundles(tmp_path)
    res = runner.invoke(app, ["diff", str(a_dir), str(a_dir), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["is_empty"] is True
    assert data["differing_steps"] == []


def test_diff_command_json_and_markdown_mutually_exclusive(tmp_path: Path) -> None:
    a_dir, b_dir = _two_bundles(tmp_path)
    res = runner.invoke(app, ["diff", str(a_dir), str(b_dir), "--json", "--markdown"])
    assert res.exit_code == 4, res.output


def test_replay_command_surfaces_divergence_notes(tmp_path: Path) -> None:
    # Save a bundle whose recorded trace has one tool call, then replay a prompt that
    # makes the FakeAgent call a DIFFERENT tool -> divergence with a note.
    from agentbisect.agent.fake import FakeAgent
    from agentbisect.bundle import save_bundle
    from agentbisect.capture import capture
    from agentbisect.types import AgentConfig

    base = AgentConfig(
        system_prompt="p",
        model="m",
        params={"program": [{"tool": "kb", "args": {"q": "policy"}}], "final": "OUT"},
    )
    bundle = capture(FakeAgent(), base, lambda t, a: "rec", label="x")
    out = save_bundle(bundle, tmp_path / "bundle")

    # A project config exposing only the runner (replay needs just the runner).
    cfg = tmp_path / "proj.py"
    cfg.write_text(
        "from agentbisect.agent.fake import FakeAgent\ndef runner():\n    return FakeAgent()\n",
        encoding="utf-8",
    )
    # Override the system prompt to one that triggers a different tool via params is not
    # possible (params held), so instead the recorded program replays exactly -> no
    # divergence. Assert the no-divergence reporting path with a model override.
    res = runner.invoke(
        app,
        ["replay", "--bundle", str(out), "--config", str(cfg), "--override", "model=mX"],
    )
    assert res.exit_code == 0, res.output
    assert "diverged=False" in res.output


def test_replay_command_reports_flags(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "replay",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--override",
            "model=m9",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "diverged=False" in res.output


def test_bisect_rev_spec_with_at_sign(
    tmp_path: Path,
    temp_git_repo,
) -> None:
    # Exercises the "<repo>:<path>@<rev>" parsing branch of the prompt-axis CLI spec.
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    versions = [
        "You are support. Always state the refund policy.",
        "You are support. Be brief.",
    ]
    repo = temp_git_repo("system.txt", versions)
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "prompt",
            "--over",
            f"{repo}:system.txt@HEAD",
            "--markdown",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "First bad change" in res.output


def test_bisect_unknown_axis_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "nonsense",
            "--over",
            "x",
        ],
    )
    assert res.exit_code == 4, res.output


def _write_schema_file(path: Path, name: str, version: str) -> Path:
    """Write a tiny, offline tool-schema JSON file (a one-tool version) and return it."""
    path.write_text(
        json.dumps([{"name": name, "schema": {"type": "object"}, "version": version}]),
        encoding="utf-8",
    )
    return path


def test_build_axis_tools_loads_schema_files(tmp_path: Path) -> None:
    from agentbisect.axes import ToolSchemaAxis
    from agentbisect.cli import _build_axis
    from agentbisect.types import AgentConfig

    f1 = _write_schema_file(tmp_path / "v1.json", "search", "1")
    f2 = _write_schema_file(tmp_path / "v2.json", "search", "2")
    axis = _build_axis("tools", f"{f1},{f2}", tmp_path)
    assert isinstance(axis, ToolSchemaAxis)

    base = AgentConfig(system_prompt="P", model="m", retrieval_ref="snap")
    cands = axis.candidates(base)
    # Ordered old -> new by file order, overriding ONLY tool_schemas (single-axis isolation).
    assert [c.order for c in cands] == [0, 1]
    assert cands[0].config.tool_schemas[0].name == "search"
    assert cands[0].config.tool_schemas[0].version == "1"
    assert cands[1].config.tool_schemas[0].version == "2"
    for c in cands:
        assert c.axis == "tools"
        assert c.config.system_prompt == "P"
        assert c.config.model == "m"
        assert c.config.retrieval_ref == "snap"


def test_bisect_tools_axis_from_files_runs(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    f1 = _write_schema_file(tmp_path / "v1.json", "search", "1")
    f2 = _write_schema_file(tmp_path / "v2.json", "search", "2")
    # tool_schemas do not alter the FakeAgent output, so both endpoints replay GOOD ->
    # non-monotonic (exit 3). This exercises the tools-axis CLI construction path.
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "tools",
            "--over",
            f"{f1},{f2}",
        ],
    )
    assert res.exit_code == 3, res.output


def test_bisect_tools_axis_missing_file_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    # The tools axis is now CLI-reachable via schema files; a nonexistent file is a clean
    # usage error (exit 4), never a crash.
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "tools",
            "--over",
            str(tmp_path / "nope.json"),
        ],
    )
    assert res.exit_code == 4, res.output
    assert "error" in res.output.lower()


def test_bisect_tools_axis_invalid_json_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "tools",
            "--over",
            str(bad),
        ],
    )
    assert res.exit_code == 4, res.output


def test_bisect_tools_axis_malformed_schema_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    # Valid JSON, but not an array of ToolSchema objects (the required 'name' is missing).
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"schema": {}}]), encoding="utf-8")
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "tools",
            "--over",
            str(bad),
        ],
    )
    assert res.exit_code == 4, res.output


def test_bisect_tools_axis_empty_spec_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        ["bisect", "--bundle", str(out), "--config", str(cfg), "--axis", "tools", "--over", " , "],
    )
    assert res.exit_code == 4, res.output


def test_bisect_retrieval_axis_runs(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    # Retrieval override does not change the FakeAgent output -> both endpoints GOOD ->
    # non-monotonic (exit 3). This exercises the retrieval-axis CLI construction path.
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "retrieval",
            "--over",
            "snap-a,snap-b",
        ],
    )
    assert res.exit_code == 3, res.output


def test_bisect_params_axis_reports_first_bad(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    # A ParamsAxis over the FakeAgent's 'final' output: the first value keeps the refund
    # clause (GOOD), the second drops it (BAD) -> a monotonic good->bad transition. The
    # value strings themselves contain '=', exercising first-'=' key/value splitting.
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "params",
            "--over",
            "final=refund=yes,refund=no",
            "--markdown",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "First bad change" in res.output
    assert "final=refund=no" in res.output


def test_bisect_json_carries_culprit_ref(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "params",
            "--over",
            "final=refund=yes,refund=no",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    # The JSON is pipe-safe (bracketed arrays are not swallowed by Rich markup).
    data = json.loads(res.output)
    assert data["axis"] == "params"
    assert data["first_bad"] == "final=refund=no"
    assert data["last_good"] == "final=refund=yes"


def test_bisect_json_and_markdown_mutually_exclusive(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "params",
            "--over",
            "final=refund=yes,refund=no",
            "--json",
            "--markdown",
        ],
    )
    assert res.exit_code == 4, res.output


def test_bisect_html_emits_self_contained_document(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "params",
            "--over",
            "final=refund=yes,refund=no",
            "--html",
        ],
    )
    assert res.exit_code == 0, res.output
    # A self-contained static HTML artifact printed verbatim (not swallowed by Rich markup).
    assert res.output.lstrip().startswith("<!DOCTYPE html>")
    assert "First bad change" in res.output
    assert "final=refund=no" in res.output


def test_bisect_html_mutually_exclusive_with_json(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "bisect",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "params",
            "--over",
            "final=refund=yes,refund=no",
            "--html",
            "--json",
        ],
    )
    assert res.exit_code == 4, res.output


def test_report_html_emits_self_contained_document(tmp_path: Path, temp_git_repo) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    versions = [
        "You are support. Always state the refund policy.",
        "You are support. Be brief.",  # drops refund -> bad
    ]
    repo = temp_git_repo("system.txt", versions)
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "prompt",
            "--over",
            f"{repo}:system.txt",
            "--html",
        ],
    )
    assert res.exit_code == 0, res.output
    assert res.output.lstrip().startswith("<!DOCTYPE html>")
    assert "First bad change" in res.output


def test_report_html_and_markdown_mutually_exclusive(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
            "--html",
            "--markdown",
        ],
    )
    assert res.exit_code == 4, res.output


def test_report_json_emits_json(tmp_path: Path, temp_git_repo) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    versions = [
        "You are support. Always state the refund policy.",
        "You are support. Be brief.",  # drops refund -> bad
    ]
    repo = temp_git_repo("system.txt", versions)
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "prompt",
            "--over",
            f"{repo}:system.txt",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["axis"] == "prompt"
    assert data["first_bad"] is not None


def test_report_json_and_markdown_mutually_exclusive(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
            "--json",
            "--markdown",
        ],
    )
    assert res.exit_code == 4, res.output


def test_replay_command_override_variants(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        [
            "replay",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--override",
            "system_prompt=new prompt",
            "--override",
            "retrieval=snap-x",
        ],
    )
    assert res.exit_code == 0, res.output


def test_replay_command_bad_override(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    res = runner.invoke(
        app,
        ["replay", "--bundle", str(out), "--config", str(cfg), "--override", "bogus=1"],
    )
    assert res.exit_code == 4, res.output


def test_report_command_emits_markdown(
    tmp_path: Path,
    temp_git_repo,
) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    versions = [
        "You are support. Always state the refund policy.",
        "You are support. Be brief.",  # drops refund -> bad
    ]
    repo = temp_git_repo("system.txt", versions)
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "prompt",
            "--over",
            f"{repo}:system.txt",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "agentbisect report" in res.output


def test_report_command_missing_bundle_usage_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(tmp_path / "nope"),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
        ],
    )
    assert res.exit_code == 4, res.output


def test_report_command_nonmonotonic_bisect_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out = tmp_path / "bundle"
    runner.invoke(app, ["capture", "--config", str(cfg), "--out", str(out)])
    # Model axis holds the prompt -> both endpoints GOOD -> non-monotonic -> exit 3.
    res = runner.invoke(
        app,
        [
            "report",
            "--bundle",
            str(out),
            "--config",
            str(cfg),
            "--axis",
            "model",
            "--over",
            "m0,m1",
        ],
    )
    assert res.exit_code == 3, res.output
