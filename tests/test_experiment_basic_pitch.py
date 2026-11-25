"""Tests for the Basic Pitch experiment builders."""

from __future__ import annotations

from pathlib import Path

from experiments.deep_learning.basic_pitch.pipeline import (
    build_features,
    build_model,
    build_parameter_groups,
)
from experiments.deep_learning.common.spec import TrainingConfig
from scribr.deep_learning.basic_pitch import (
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


def test_parameter_groups_cover_trunk_and_head() -> None:
    config = TrainingConfig.load(CONFIG_PATH)
    model = build_model(config.model, config.features)
    groups = build_parameter_groups(model, config.optimization)

    assert [group["lr"] for group in groups] == [
        config.optimization.learning_rate,
        config.optimization.head_learning_rate,
    ]
    grouped = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]

    model_parameters = list(model.parameters())

    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model_parameters
    }
    assert len(grouped) == len(model_parameters)
