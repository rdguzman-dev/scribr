"""Tests for the synthesis-on-demand training dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scribr.deep_learning import (
    LogMelSpectrogram,
    SynthesizedMelodyDataset,
    pitch_sequence_to_classes,
)
from scribr.representation import (
    HOLD_TOKEN,
    NOTE_OFF_TOKEN,
    Melody,
    MelodyExample,
    Note,
)

SEQUENCE = (60, HOLD_TOKEN, HOLD_TOKEN, NOTE_OFF_TOKEN)


class FakeSynthesizer:
    def __init__(self, sample_rate: int = 22_050) -> None:
        self.sample_rate = sample_rate
        self.calls = 0

    def synthesize(self, melody: Melody) -> np.ndarray:
        self.calls += 1

        return np.full(4096, 0.1, dtype=np.float32)


class CountingFactory:
    def __init__(self) -> None:
        self.created = 0

    def __call__(self) -> FakeSynthesizer:
        self.created += 1

        return FakeSynthesizer()


def tiny_features() -> LogMelSpectrogram:
    return LogMelSpectrogram(
        sample_rate=22_050,
        n_fft=256,
        hop_length=64,
        n_mels=16,
        f_min=27.5,
        f_max=8_000.0,
        top_db=80.0,
        num_frames=16,
    )


def make_example() -> MelodyExample:
    return MelodyExample(
        melody=Melody((Note(60, 0.0, 0.75),)),
        pitch_sequence=SEQUENCE,
    )


def test_item_features_and_target() -> None:
    dataset = SynthesizedMelodyDataset(
        [make_example()],
        CountingFactory(),
        tiny_features(),
    )

    features, target = dataset[0]

    assert len(dataset) == 1
    assert features.shape == (16, 16)
    assert target.dtype == torch.long
    assert target.tolist() == list(pitch_sequence_to_classes(SEQUENCE))


def test_mismatched_sample_rates_raise() -> None:
    dataset = SynthesizedMelodyDataset(
        [make_example()],
        lambda: FakeSynthesizer(sample_rate=16_000),
        tiny_features(),
    )

    with pytest.raises(ValueError, match="sample rate"):
        dataset[0]


def test_synthesizer_is_built_once_per_dataset() -> None:
    factory = CountingFactory()
    dataset = SynthesizedMelodyDataset(
        [make_example(), make_example()],
        factory,
        tiny_features(),
    )

    dataset[0]
    dataset[1]

    assert factory.created == 1
    assert factory() is not None
