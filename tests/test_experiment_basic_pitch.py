"""Tests for the Basic Pitch experiment builders."""

from __future__ import annotations

from pathlib import Path

from experiments.deep_learning.basic_pitch.pipeline import (
    build_features,
    build_model,
)
from experiments.deep_learning.common.spec import TrainingConfig
from scribr.deep_learning import (
    BasicPitchFrontend,
    BasicPitchPitchSequenceCNN,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deep_learning"
    / "basic_pitch"
    / "config.json"
)


def test_committed_config_builds_features_and_model() -> None:
    config = TrainingConfig.load(CONFIG_PATH)
    features = build_features(config.features)
    model = build_model(config.model, config.features)

    assert isinstance(features, BasicPitchFrontend)
    assert features.sample_rate == config.synthesis.sample_rate
    assert features.num_frames is None
    assert isinstance(model, BasicPitchPitchSequenceCNN)
    assert model.steps == 64
