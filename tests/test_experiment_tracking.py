"""Tests for local MLflow experiment tracking."""

from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from experiments.common.tracking import (
    log_artifact,
    log_metrics,
    log_params,
    start_run,
)


def test_run_records_params_metrics_and_artifacts(tmp_path: Path) -> None:
    tracking_uri = tmp_path / "mlruns"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "loss.txt").write_text("loss curve\n", encoding="utf-8")

    with start_run(
        "deep_learning",
        run_name="baseline",
        tracking_uri=tracking_uri,
        tags={"stage": "test"},
    ) as run:
        log_params(
            {
                "learning_rate": 0.001,
                "batch_size": 16,
                "shuffle": True,
                "instruments": ("piano", "violin"),
            }
        )
        log_metrics({"train_loss": 1.5, "val_loss": 1.2}, step=0)
        log_metrics({"train_loss": 0.8, "val_loss": 0.9}, step=1)
        log_artifact(checkpoint)
        log_artifact(figures)

    client = MlflowClient(tracking_uri=tracking_uri.as_uri())
    stored = client.get_run(run.info.run_id)

    assert client.get_experiment(stored.info.experiment_id).name == "deep_learning"
    assert stored.info.run_name == "baseline"
    assert stored.data.tags["stage"] == "test"
    assert stored.data.params == {
        "learning_rate": "0.001",
        "batch_size": "16",
        "shuffle": "True",
        "instruments": "('piano', 'violin')",
    }
    assert stored.data.metrics == {"train_loss": 0.8, "val_loss": 0.9}

    root_artifacts = client.list_artifacts(run.info.run_id)
    assert {entry.path for entry in root_artifacts} == {
        "checkpoint.pt",
        "figures",
    }
    figure_files = client.list_artifacts(run.info.run_id, "figures")
    assert {entry.path for entry in figure_files} == {"figures/loss.txt"}

    downloaded = client.download_artifacts(run.info.run_id, "checkpoint.pt")
    assert Path(downloaded).read_bytes() == b"weights"
