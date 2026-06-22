"""RunBundle (de)serialization with an integrity hash and version validation.

A bundle is stored as a directory containing a single ``bundle.json`` file (a directory
is used so future additions -- e.g. a retrieval snapshot -- can live alongside it without
breaking the format). The ``integrity`` field is a hash of ``config`` + ``trace`` so any
tampering with the stored bundle is detected on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .canonical import stable_hash
from .types import BUNDLE_VERSION, AgentConfig, RunBundle, Step, Trace

__all__ = [
    "BUNDLE_FILENAME",
    "BundleError",
    "BundleIntegrityError",
    "BundleVersionError",
    "compute_integrity",
    "load_bundle",
    "make_bundle",
    "save_bundle",
]

#: Name of the JSON file inside a bundle directory.
BUNDLE_FILENAME = "bundle.json"

_STEP_LIST_ADAPTER: TypeAdapter[list[Step]] = TypeAdapter(list[Step])


class BundleError(Exception):
    """Base class for bundle (de)serialization errors."""


class BundleVersionError(BundleError):
    """Raised when a bundle's schema version is not understood by this library."""


class BundleIntegrityError(BundleError):
    """Raised when a bundle's stored integrity hash does not match its contents."""


def compute_integrity(config: AgentConfig, trace: Trace) -> str:
    """Return the integrity hash over a config + trace."""
    return stable_hash(
        {
            "config": config.fingerprint(),
            "trace": trace.digest(),
        }
    )


def make_bundle(*, config: AgentConfig, trace: Trace, label: str) -> RunBundle:
    """Build a :class:`RunBundle` with a freshly computed integrity hash."""
    return RunBundle(
        version=BUNDLE_VERSION,
        config=config,
        trace=trace,
        label=label,
        integrity=compute_integrity(config, trace),
    )


def _bundle_to_dict(bundle: RunBundle) -> dict[str, Any]:
    return {
        "version": bundle.version,
        "label": bundle.label,
        "integrity": bundle.integrity,
        "config": bundle.config.model_dump(mode="json", by_alias=True),
        "trace": {
            "steps": [s.model_dump(mode="json") for s in bundle.trace.steps],
            "final_output": bundle.trace.final_output,
        },
    }


def save_bundle(bundle: RunBundle, path: str | Path) -> Path:
    """Write ``bundle`` to the directory ``path`` and return the directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / BUNDLE_FILENAME
    file_path.write_text(
        json.dumps(_bundle_to_dict(bundle), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return directory


def load_bundle(path: str | Path) -> RunBundle:
    """Load and validate a bundle from a directory (or a direct ``bundle.json`` path).

    Raises
    ------
    BundleVersionError
        If the stored version is newer than this library understands.
    BundleIntegrityError
        If the recomputed integrity hash does not match the stored one.
    """
    p = Path(path)
    file_path = p / BUNDLE_FILENAME if p.is_dir() else p
    if not file_path.exists():
        raise BundleError(f"bundle not found: {p}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read bundle at {p}: {exc}") from exc

    version = int(data["version"])
    if version > BUNDLE_VERSION:
        raise BundleVersionError(
            f"bundle version {version} is newer than supported version {BUNDLE_VERSION}; "
            "upgrade agentbisect to read it"
        )

    config = AgentConfig.model_validate(data["config"])
    steps = _STEP_LIST_ADAPTER.validate_python(data["trace"]["steps"])
    trace = Trace(steps=tuple(steps), final_output=data["trace"].get("final_output", ""))

    expected = compute_integrity(config, trace)
    stored = str(data.get("integrity", ""))
    if stored != expected:
        raise BundleIntegrityError(
            "bundle integrity check failed: contents do not match the stored hash "
            "(the bundle may be corrupted or tampered with)"
        )

    return RunBundle(
        version=version,
        config=config,
        trace=trace,
        label=str(data.get("label", "")),
        integrity=stored,
    )
