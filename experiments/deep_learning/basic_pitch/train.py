"""Train the Basic Pitch transfer model and track the run with MLflow.

One run fine-tunes the pretrained contour trunk with the project's
per-step 90-class head. Training audio is synthesized on the fly from the
piano program by default and the model is scored every epoch on a fixed
validation slice, with the full `mir_eval` suite replayed on every
evaluation instrument.
"""

from __future__ import annotations

from pathlib import Path

from experiments.deep_learning.common.cli import train_main

from .pipeline import (
    build_features,
    build_model,
    build_parameter_groups,
)

_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    train_main(
        build_features,
        build_model,
        default_config=_DIR / "config.json",
        default_artifacts_dir=_DIR / "artifacts",
        parameter_groups=build_parameter_groups,
        description=__doc__,
    )
