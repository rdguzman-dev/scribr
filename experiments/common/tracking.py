"""Local MLflow tracking for experiment runs.

Run metadata is written to `mlflow.db` at the repository root and never
leaves the machine. Artifacts go to `mlruns/` next to the database. Point
`tracking_uri` at another SQLite database to isolate runs.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Set before importing mlflow: the assistant hint would otherwise print
# once per `DataLoader` worker process.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")

import mlflow  # noqa: E402

# Repository root, used for the default store and git metadata tags.
_REPOSITORY = Path(__file__).resolve().parents[2]

# `<repo>/mlflow.db`, the default local SQLite tracking store.
DEFAULT_TRACKING_URI = f"sqlite:///{_REPOSITORY / 'mlflow.db'}"


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
        tracking_uri: Tracking store URI, or a path to a SQLite database.
            Defaults to `mlflow.db` at the repository root.
        tags: Run tags, such as the model or dataset variant.

    Yields:
        The active `mlflow.ActiveRun`.
    """
    resolved_uri = _as_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    experiment_id = _ensure_experiment(
        experiment, _default_artifact_root(resolved_uri)
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=dict(tags) if tags is not None else None,
    ) as run:
        yield run


def _ensure_experiment(name: str, artifact_location: str | None) -> str:
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(name)

    if existing is not None:
        return existing.experiment_id

    return client.create_experiment(name, artifact_location=artifact_location)


def _default_artifact_root(tracking_uri: str) -> str | None:
    """Return the artifact root for an experiment on this store.

    SQLite stores keep a `mlruns/` directory beside the database, which
    also keeps test runs inside their temporary directory. Other stores
    fall back to MLflow's artifact root.
    """
    if not tracking_uri.startswith("sqlite:///"):
        return None

    database = Path(tracking_uri.removeprefix("sqlite:///"))
    return (database.expanduser().resolve().parent / "mlruns").as_uri()


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


def _as_tracking_uri(tracking_uri: str | Path) -> str:
    """Return a tracking URI, treating plain paths as SQLite databases."""
    if isinstance(tracking_uri, str) and "://" in tracking_uri:
        return tracking_uri
    return f"sqlite:///{Path(tracking_uri).expanduser().resolve()}"
