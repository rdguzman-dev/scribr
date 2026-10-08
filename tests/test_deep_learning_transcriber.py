"""Tests for the deep learning transcriber."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scribr.deep_learning import (
    NUM_CLASSES,
    DeepLearningTranscriber,
    LogMelSpectrogram,
    pitch_sequence_to_classes,
)
from scribr.representation import (
    HOLD_TOKEN,
    NOTE_OFF_TOKEN,
    Melody,
    decode_pitch_sequence,
)


class StubModel(nn.Module):
    """Return fixed logits that decode to a known token sequence."""

    def __init__(self, sequence: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer(
            "classes",
            torch.tensor(
                pitch_sequence_to_classes(sequence), dtype=torch.long
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        steps = self.classes.numel()
        logits = torch.zeros(batch, steps, NUM_CLASSES)
        index = self.classes.view(1, steps, 1).expand(batch, steps, 1)

        return logits.scatter(2, index, 1.0)


def tiny_features() -> LogMelSpectrogram:
    return LogMelSpectrogram(
        sample_rate=22_050,
        n_fft=256,
        hop_length=64,
        n_mels=16,
        f_min=27.5,
        f_max=8_000.0,
        top_db=80.0,
        num_frames=8,
    )


def stub_transcriber(sequence: tuple[int, ...]) -> DeepLearningTranscriber:
    return DeepLearningTranscriber(
        StubModel(sequence),
        tiny_features(),
        training_tempo=120.0,
        device="cpu",
    )


def test_predictions_decode_through_the_codec() -> None:
    sequence = (60, HOLD_TOKEN, 62, NOTE_OFF_TOKEN)
    transcriber = stub_transcriber(sequence)
    audio = np.zeros(4096, dtype=np.float32)

    melody = transcriber.transcribe(audio, 22_050, 120.0)

    assert melody == Melody(decode_pitch_sequence(sequence))


def test_empty_audio_transcribes_to_an_empty_melody() -> None:
    transcriber = stub_transcriber((60, NOTE_OFF_TOKEN))

    assert transcriber.transcribe(np.zeros(0), 22_050, 120.0) == Melody(())


def test_model_is_switched_to_eval_mode() -> None:
    model = StubModel((60, NOTE_OFF_TOKEN))
    model.train()

    DeepLearningTranscriber(model, tiny_features(), device="cpu")

    assert not model.training


@pytest.mark.parametrize(
    ("sample_rate", "tempo"),
    [(22_050, 60.0), (44_100, 120.0), (16_000, 90.0)],
)
def test_sample_rate_and_tempo_mismatches_are_handled(
    sample_rate: int,
    tempo: float,
) -> None:
    transcriber = stub_transcriber((60, NOTE_OFF_TOKEN))
    audio = np.zeros(sample_rate, dtype=np.float32)

    melody = transcriber.transcribe(audio, sample_rate, tempo)

    assert isinstance(melody, Melody)


def test_invalid_training_tempo_raises() -> None:
    with pytest.raises(ValueError):
        DeepLearningTranscriber(
            StubModel((60,)),
            tiny_features(),
            training_tempo=0.0,
        )


def test_invalid_tempo_raises() -> None:
    transcriber = stub_transcriber((60,))

    with pytest.raises(ValueError):
        transcriber.transcribe(np.zeros(16, dtype=np.float32), 22_050, 0.0)
