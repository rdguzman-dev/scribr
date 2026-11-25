"""Feature and model construction for the Basic Pitch experiment."""

from __future__ import annotations

from typing import Any, Mapping

from experiments.deep_learning.common.spec import OptimizationSpec
from scribr.deep_learning.basic_pitch import (
    BasicPitchFrontend,
    BasicPitchPitchSequenceCNN,
)
from scribr.representation import STEPS_PER_MELODY


def build_features(spec: Mapping[str, Any]) -> BasicPitchFrontend:
    """Build the fixed Basic Pitch harmonic CQT extractor."""
    return BasicPitchFrontend(
        sample_rate=spec["sample_rate"],
        num_frames=spec["num_frames"],
    )


def build_model(
    spec: Mapping[str, Any],
    features: Mapping[str, Any],
) -> BasicPitchPitchSequenceCNN:
    """Build the trunk plus per-step head described by `spec`."""
    return BasicPitchPitchSequenceCNN(
        steps=STEPS_PER_MELODY,
        dropout=spec["dropout"],
        pretrained=spec["pretrained"],
    )


def build_parameter_groups(
    model: BasicPitchPitchSequenceCNN,
    optimization: OptimizationSpec,
) -> list[dict[str, Any]]:
    """Give the new head a faster rate than the pretrained trunk."""
    if optimization.head_learning_rate is None:
        return [
            {
                "params": list(model.parameters()),
                "lr": optimization.learning_rate,
            }
        ]

    return [
        {
            "params": list(model.trunk.parameters()),
            "lr": optimization.learning_rate,
        },
        {
            "params": [
                *model.temporal.parameters(),
                *model.classifier.parameters(),
            ],
            "lr": optimization.head_learning_rate,
        },
    ]
