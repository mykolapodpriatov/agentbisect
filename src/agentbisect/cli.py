"""The ``agentbisect`` command-line interface (typer).

Commands:

* ``capture``  -- capture a run into a bundle directory.
* ``bisect``   -- bisect an axis over a bundle; prints first-bad + axis + repro + diff.
* ``replay``   -- replay a bundle under an axis override and report divergence flags.
* ``diff``     -- behavioral diff between two bundles.
* ``report``   -- re-render a bisection report from a bundle + axis (alias of bisect's report).

Exit codes: ``0`` success / first-bad found, ``2`` ambiguous range, ``3`` a bisect
precondition failed (untestable endpoint / non-monotonic), ``4`` a usage/config error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from .axes import Axis, ModelListAxis, PromptGitAxis, RetrievalAxis
from .bisect import NonMonotonicError, UntestableEndpointError
from .bundle import BundleError, load_bundle, save_bundle
from .capture import capture
from .config import ConfigError, load_project_config
from .diff import diff as diff_traces
from .driver import run_bisection
from .mock_tools import DivergencePolicy
from .report import render_markdown, render_rich

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="git-bisect for LLM-agent regressions.",
)
console = Console()
err_console = Console(stderr=True)

EXIT_OK = 0
EXIT_AMBIGUOUS = 2
EXIT_BISECT_ERROR = 3
EXIT_USAGE = 4


def _fail(message: str, code: int) -> NoReturn:
    """Print an error and exit with ``code`` (never returns)."""
    err_console.print(f"[red]error:[/] {message}")
    raise typer.Exit(code)


@app.command(name="capture")
def capture_cmd(
    config: Annotated[Path, typer.Option("--config", help="Project config .py file.")],
    out: Annotated[Path, typer.Option("--out", help="Output bundle directory.")],
    label: Annotated[str, typer.Option("--label", help="Caller-supplied bundle label.")] = "",
) -> None:
    """Capture an agent run into a versioned bundle directory."""
    try:
        project = load_project_config(config)
        runner = project.runner()
        agent_config = project.config()
        executor = project.tool_executor()
    except ConfigError as exc:
        _fail(str(exc), EXIT_USAGE)
    bundle = capture(runner, agent_config, executor, label=label)
    save_bundle(bundle, out)
    console.print(f"[green]captured[/] {len(bundle.trace.steps)} step(s) -> {out}")


def _build_axis(axis: str, over: str, bundle_dir: Path) -> Axis:
    """Construct an axis provider from the ``--axis`` / ``--over`` CLI options."""
    if axis == "model":
        models = [m.strip() for m in over.split(",") if m.strip()]
        return ModelListAxis(models)
    if axis == "prompt":
        # over = "<repo>:<path>[@rev]"
        repo, _, rest = over.partition(":")
        path, _, rev = rest.partition("@")
        return PromptGitAxis(repo or ".", path, rev or "HEAD")
    if axis == "retrieval":
        snaps = [s.strip() for s in over.split(",") if s.strip()]
        return RetrievalAxis(snaps)
    if axis == "tools":
        raise ConfigError(
            "the 'tools' axis requires schema objects; drive it via the library "
            "(ToolSchemaAxis) rather than the CLI string form"
        )
    raise ConfigError(f"unknown axis {axis!r}; expected model|prompt|tools|retrieval")


@app.command()
def bisect(
    bundle: Annotated[Path, typer.Option("--bundle", help="Bundle directory.")],
    config: Annotated[Path, typer.Option("--config", help="Project config .py (for the oracle).")],
    axis: Annotated[str, typer.Option("--axis", help="Axis: model|prompt|tools|retrieval.")],
    over: Annotated[str, typer.Option("--over", help="Axis spec (see docs).")],
    policy: Annotated[
        DivergencePolicy, typer.Option("--policy", help="Divergence policy.")
    ] = DivergencePolicy.SKIP,
    markdown: Annotated[bool, typer.Option("--markdown", help="Emit Markdown instead.")] = False,
) -> None:
    """Bisect an axis over a captured bundle and report the first bad change."""
    try:
        run_bundle = load_bundle(bundle)
        project = load_project_config(config)
        runner = project.runner()
        oracle = project.oracle()
        axis_provider = _build_axis(axis, over, bundle)
        candidates = axis_provider.candidates(run_bundle.config)
    except (BundleError, ConfigError, ValueError) as exc:
        _fail(str(exc), EXIT_USAGE)

    try:
        outcome = run_bisection(runner, run_bundle, candidates, oracle, policy=policy)
    except (UntestableEndpointError, NonMonotonicError) as exc:
        _fail(str(exc), EXIT_BISECT_ERROR)

    if markdown:
        console.print(render_markdown(outcome))
    else:
        render_rich(outcome, console)

    raise typer.Exit(EXIT_OK if outcome.result.first_bad is not None else EXIT_AMBIGUOUS)


@app.command()
def replay(
    bundle: Annotated[Path, typer.Option("--bundle", help="Bundle directory.")],
    config: Annotated[Path, typer.Option("--config", help="Project config .py (for the runner).")],
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="axis=value override, e.g. model=gpt-x."),
    ] = None,
    policy: Annotated[
        DivergencePolicy, typer.Option("--policy", help="Divergence policy.")
    ] = DivergencePolicy.SKIP,
) -> None:
    """Replay a bundle once under axis overrides and print divergence flags."""
    from .replay import replay as do_replay

    try:
        run_bundle = load_bundle(bundle)
        project = load_project_config(config)
        runner = project.runner()
    except (BundleError, ConfigError) as exc:
        _fail(str(exc), EXIT_USAGE)

    changes: dict[str, object] = {}
    for item in override or []:
        key, _, value = item.partition("=")
        key = key.strip()
        if key == "model":
            changes["model"] = value
        elif key == "retrieval":
            changes["retrieval_ref"] = value
        elif key == "system_prompt":
            changes["system_prompt"] = value
        else:
            _fail(f"unsupported override {key!r}", EXIT_USAGE)

    candidate_config = run_bundle.config.with_overrides(**changes)
    result = do_replay(runner, candidate_config, run_bundle.trace, policy=policy)
    console.print(
        f"diverged={result.diverged} "
        f"nearest={result.has_nearest_substitutions} "
        f"passthrough={result.used_passthrough}"
    )
    console.print(f"final: {result.trace.final_output!r}")
    for note in result.notes:
        console.print(f"  - {note}")


@app.command()
def diff(
    a: Annotated[Path, typer.Argument(help="First bundle directory.")],
    b: Annotated[Path, typer.Argument(help="Second bundle directory.")],
) -> None:
    """Print a step-aligned behavioral diff between two bundles' traces."""
    try:
        bundle_a = load_bundle(a)
        bundle_b = load_bundle(b)
    except BundleError as exc:
        _fail(str(exc), EXIT_USAGE)
    bdiff = diff_traces(bundle_a.trace, bundle_b.trace)
    if bdiff.is_empty:
        console.print("[green]no behavioral difference[/]")
        return
    console.print(f"first divergence at step: {bdiff.first_divergence}")
    for sd in bdiff.steps:
        if not sd.same:
            console.print(f"  step {sd.index}: {sd.left}  =>  {sd.right}")
    if bdiff.final_output_changed:
        console.print(f"final: {bdiff.left_final!r}  =>  {bdiff.right_final!r}")


@app.command()
def report(
    bundle: Annotated[Path, typer.Option("--bundle", help="Bundle directory.")],
    config: Annotated[Path, typer.Option("--config", help="Project config .py.")],
    axis: Annotated[str, typer.Option("--axis", help="Axis: model|prompt|tools|retrieval.")],
    over: Annotated[str, typer.Option("--over", help="Axis spec (see docs).")],
) -> None:
    """Run a bisection and emit a Markdown culprit report."""
    try:
        run_bundle = load_bundle(bundle)
        project = load_project_config(config)
        runner = project.runner()
        oracle = project.oracle()
        candidates = _build_axis(axis, over, bundle).candidates(run_bundle.config)
    except (BundleError, ConfigError, ValueError) as exc:
        _fail(str(exc), EXIT_USAGE)
    try:
        outcome = run_bisection(runner, run_bundle, candidates, oracle)
    except (UntestableEndpointError, NonMonotonicError) as exc:
        _fail(str(exc), EXIT_BISECT_ERROR)
    console.print(render_markdown(outcome))


if __name__ == "__main__":  # pragma: no cover
    app()
