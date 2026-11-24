"""Feature and model construction for the Basic Pitch experiment."""

from __future__ import annotations

from typing import Any, Mapping

from scribr.deep_learning import (
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
