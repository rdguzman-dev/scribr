"""Evaluate a trained checkpoint on a dataset split.

Loads the config from the checkpoint, so the model and features are
rebuilt exactly as they were trained. Scoring covers every evaluation
instrument and writes `report.json` and `report.md` next to the
checkpoint, then records the summary in a separate MLflow run tagged
with the split.
"""

from __future__ import annotations

from experiments.deep_learning.common.cli import evaluate_main

from .pipeline import build_features, build_model

if __name__ == "__main__":
    evaluate_main(
        build_features,
        build_model,
        description=__doc__,
    )
