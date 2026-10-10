"""Tests for the log-mel CNN experiment builders."""

from __future__ import annotations

from pathlib import Path

from experiments.deep_learning.common.spec import TrainingConfig
from experiments.deep_learning.logmel_cnn.data import (
    build_features,
    build_model,
)
from scribr.deep_learning import LogMelSpectrogram, PitchSequenceCNN

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deep_learning"
    / "logmel_cnn"
    / "config.json"
)


def test_committed_config_builds_features_and_model() -> None:
    config = TrainingConfig.load(CONFIG_PATH)
    features = build_features(config.features)
    model = build_model(config.model, config.features)

    assert isinstance(features, LogMelSpectrogram)
    assert features.sample_rate == config.synthesis.sample_rate
    assert features.n_mels == config.features["n_mels"]
    assert isinstance(model, PitchSequenceCNN)
    assert model.channels == tuple(config.model["channels"])
