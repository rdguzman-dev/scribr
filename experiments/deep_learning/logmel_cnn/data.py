"""Feature and model construction for the log-mel CNN experiment."""

from __future__ import annotations

from typing import Any, Mapping

from scribr.deep_learning import LogMelSpectrogram, PitchSequenceCNN
from scribr.representation import STEPS_PER_MELODY


def build_features(spec: Mapping[str, Any]) -> LogMelSpectrogram:
    """Build the feature extractor described by `spec`."""
    return LogMelSpectrogram(**spec)


def build_model(
    spec: Mapping[str, Any],
    features: Mapping[str, Any],
) -> PitchSequenceCNN:
    """Build the model described by `spec` for `features["n_mels"]`."""
    return PitchSequenceCNN(
        n_mels=features["n_mels"],
        channels=spec["channels"],
        steps=STEPS_PER_MELODY,
        dropout=spec["dropout"],
    )
