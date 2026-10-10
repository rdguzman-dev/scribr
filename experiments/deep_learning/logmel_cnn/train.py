"""Train the log-mel CNN and track the run with MLflow.

One run trains one model. Training audio is synthesized on the fly from
the piano program by default and the model is scored every epoch on a
fixed validation slice, with the full `mir_eval` suite replayed on every
evaluation instrument. Checkpoints and the epoch history are written to
the run's artifacts directory and logged to MLflow.
"""

from __future__ import annotations

from pathlib import Path

from experiments.deep_learning.common.cli import train_main

from .data import build_features, build_model

_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    train_main(
        build_features,
        build_model,
        default_config=_DIR / "config.json",
        default_artifacts_dir=_DIR / "artifacts",
        description=__doc__,
    )
