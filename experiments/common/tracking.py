"""Local MLflow tracking for experiment runs.

Runs are written to `mlruns/` at the repository root and never leave the
machine. Point `tracking_uri` at another directory to isolate a run.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

# The file store needs an explicit opt-in; runs stay local and `mlruns/` is
# gitignored.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# `<repo>/mlruns`, the default local MLflow store.
DEFAULT_TRACKING_URI = Path(__file__).resolve().parents[2] / "mlruns"


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


def _as_uri(tracking_uri: str | Path) -> str:
    """Return a file URI for a local tracking directory."""
    return Path(tracking_uri).expanduser().resolve().as_uri()
