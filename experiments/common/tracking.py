"""Local MLflow tracking for experiment runs.

Runs are written to `mlruns/` at the repository root and never leave the
machine. Point `tracking_uri` at another directory to isolate a run.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Set before importing mlflow: it reads both of these during import. The
# file store needs an explicit opt-in, and the assistant hint would print
# once per `DataLoader` worker process.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")

import mlflow  # noqa: E402

# Repository root, used for the default store and git metadata tags.
_REPOSITORY = Path(__file__).resolve().parents[2]

# `<repo>/mlruns`, the default local MLflow store.
DEFAULT_TRACKING_URI = _REPOSITORY / "mlruns"


@contextmanager
def start_run(
    experiment: str,
    *,
    run_name: str | None = None,
    tracking_uri: str | Path = DEFAULT_TRACKING_URI,
    tags: Mapping[str, str] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Track one training or evaluation run against a local store.

    Args:
        experiment: Experiment name. Created on first use.
        run_name: Run name. MLflow generates one when omitted.
        tracking_uri: Local directory for the MLflow store. Defaults to
            `mlruns/` at the repository root.
        tags: Run tags, such as the model or dataset variant.

    Yields:
        The active `mlflow.ActiveRun`.
    """
    mlflow.set_tracking_uri(_as_uri(tracking_uri))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(
        run_name=run_name,
        tags=dict(tags) if tags is not None else None,
    ) as run:
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    """Log parameters, converting each value to a string."""
    mlflow.log_params({name: str(value) for name, value in params.items()})


def log_metrics(
    metrics: Mapping[str, float],
    *,
    step: int | None = None,
) -> None:
    """Log metrics, optionally against a training step."""
    mlflow.log_metrics(dict(metrics), step=step)


def log_artifact(path: str | Path) -> None:
    """Log a file or directory to the active run."""
    mlflow.log_artifact(str(path))


def git_metadata() -> dict[str, str]:
    """Return git commit and working-tree state as run tags.

    Returns `unknown` and `unknown` when the checkout cannot be
    inspected, so logging never fails on a machine without git.
    """
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])

    return {
        "git.commit": commit if commit else "unknown",
        "git.dirty": str(bool(status)).lower() if status is not None else "unknown",
    }


def _git(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
            cwd=_REPOSITORY,
        )

    except OSError:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _as_uri(tracking_uri: str | Path) -> str:
    """Return a file URI for a local tracking directory."""
    return Path(tracking_uri).expanduser().resolve().as_uri()
