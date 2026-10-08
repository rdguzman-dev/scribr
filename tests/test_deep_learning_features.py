"""Tests for log-mel feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scribr.deep_learning import LogMelSpectrogram, fix_length


def make_audio(seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)

    return rng.normal(0, 0.1, round(seconds * 22_050)).astype(np.float32)


def tiny_features(**overrides) -> LogMelSpectrogram:
    defaults = {
        "sample_rate": 22_050,
        "n_fft": 256,
        "hop_length": 64,
        "n_mels": 16,
        "f_min": 27.5,
        "f_max": 8_000.0,
        "top_db": 80.0,
        "num_frames": 32,
    }
    defaults.update(overrides)

    return LogMelSpectrogram(**defaults)


def test_features_are_fixed_length() -> None:
    features = tiny_features()
    result = features(make_audio())

    assert result.shape == (16, 32)
    assert result.dtype == torch.float32


def test_spectrogram_is_standardized() -> None:
    features = tiny_features(num_frames=None)
    result = features(make_audio())

    assert abs(float(result.mean())) < 1e-5
    assert abs(float(result.std()) - 1.0) < 1e-4


def test_spectrogram_keeps_natural_length() -> None:
    features = tiny_features(num_frames=None)
    result = features.spectrogram(make_audio())

    assert result.shape == (16, 1 + 22_050 // 64)


def test_extraction_is_deterministic() -> None:
    features = tiny_features()
    audio = make_audio()

    assert torch.equal(features(audio), features(audio))


def test_empty_audio_is_zero_features() -> None:
    features = tiny_features()
    result = features(make_audio()[:0])

    assert result.shape == (16, 32)
    assert torch.count_nonzero(result) == 0


def test_fix_length_pads_and_crops() -> None:
    source = torch.ones(3, 5)

    padded = fix_length(source, 8)
    assert padded.shape == (3, 8)
    assert torch.all(padded[:, 5:] == 0)
    assert torch.all(padded[:, :5] == 1)

    assert fix_length(source, 2).shape == (3, 2)
    assert fix_length(source, 5) is source


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_rate": 0},
        {"n_fft": 0},
        {"hop_length": 0},
        {"hop_length": 512},
        {"n_mels": 0},
        {"f_min": -1.0},
        {"f_max": 12_000.0},
        {"top_db": 0.0},
        {"num_frames": 0},
    ],
)
def test_invalid_features_parameters_raise(overrides: dict) -> None:
    with pytest.raises(ValueError):
        tiny_features(**overrides)
