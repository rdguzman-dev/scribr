"""Tests for the Basic Pitch harmonic CQT frontend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scribr.deep_learning.basic_pitch import (
    HARMONIC_SHIFTS,
    N_BINS,
    N_HARMONICS,
    N_STACK_BINS,
    BasicPitchFrontend,
)


@pytest.fixture(scope="module")
def frontend() -> BasicPitchFrontend:
    return BasicPitchFrontend()


def make_audio(seconds: float = 2.0) -> np.ndarray:
    samples = np.arange(round(seconds * 22_050))
    return (
        0.3 * np.sin(2 * np.pi * 440 * samples / 22_050)
    ).astype(np.float32)


def test_has_no_trainable_parameters(
    frontend: BasicPitchFrontend,
) -> None:
    assert list(frontend.parameters()) == []

    for name, buffer in frontend.named_buffers():
        assert not buffer.requires_grad, name


def test_output_shape(frontend: BasicPitchFrontend) -> None:
    result = frontend(make_audio())

    assert result.shape == (N_HARMONICS, N_STACK_BINS, 173)
    assert result.dtype == torch.float32


@pytest.mark.parametrize(
    ("samples", "frames"),
    [
        (43_844, 172),
        (8 * 22_050, 690),
        (round(8.5 * 22_050), 733),
    ],
)
def test_frame_counts(
    frontend: BasicPitchFrontend,
    samples: int,
    frames: int,
) -> None:
    result = frontend(np.zeros(samples, dtype=np.float32))

    assert result.shape == (N_HARMONICS, N_STACK_BINS, frames)


def test_harmonic_offsets() -> None:
    assert HARMONIC_SHIFTS == (-36, 0, 36, 57, 72, 84, 93, 101)


def test_frequency_shifts_zero_fill_instead_of_wrapping() -> None:
    source = torch.arange(1, 6, dtype=torch.float32).reshape(1, 5, 1)

    down = BasicPitchFrontend._shift_frequency(source, 2)
    up = BasicPitchFrontend._shift_frequency(source, -2)

    assert down[0, :, 0].tolist() == [3.0, 4.0, 5.0, 0.0, 0.0]
    assert up[0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_harmonic_stack_keeps_the_unshifted_channel() -> None:
    normalized = torch.arange(
        N_BINS,
        dtype=torch.float32,
    ).reshape(1, N_BINS, 1)
    stack = BasicPitchFrontend._harmonic_stack(normalized)

    assert stack.shape == (1, N_HARMONICS, N_STACK_BINS, 1)
    assert torch.equal(
        stack[:, 1],
        normalized[:, :N_STACK_BINS],
    )


def test_440_hz_peaks_at_the_expected_bin(
    frontend: BasicPitchFrontend,
) -> None:
    waveform = torch.from_numpy(make_audio()).unsqueeze(0)
    magnitude = frontend._cqt_magnitude(waveform)
    normalized = frontend._normalized_log(magnitude)
    peak = int(normalized.mean(dim=-1).argmax(dim=-1)[0])

    # 36 * log2(440 / 27.5) = 144.
    assert abs(peak - 144) <= 1


def test_extraction_is_deterministic(
    frontend: BasicPitchFrontend,
) -> None:
    audio = make_audio()

    assert torch.equal(frontend(audio), frontend(audio))


def test_empty_audio_is_zero_features(
    frontend: BasicPitchFrontend,
) -> None:
    result = frontend(make_audio()[:0])

    assert result.shape == (N_HARMONICS, N_STACK_BINS, 0)
    assert torch.count_nonzero(result) == 0


def test_batch_input_keeps_the_batch_axis(
    frontend: BasicPitchFrontend,
) -> None:
    result = frontend(make_audio()[None, :].repeat(2, axis=0))

    assert result.shape == (2, N_HARMONICS, N_STACK_BINS, 173)


def test_sample_rate_must_match_the_pretrained_model() -> None:
    with pytest.raises(ValueError):
        BasicPitchFrontend(sample_rate=44_100)


def test_invalid_num_frames_raises() -> None:
    with pytest.raises(ValueError):
        BasicPitchFrontend(num_frames=0)
