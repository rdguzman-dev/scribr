"""Tests for optional WAV export, reading files back with `wave`."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from scribr.wav import write_wav


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV file into samples and a sample rate."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    return np.frombuffer(frames, dtype="<i2"), sample_rate


def test_write_wav_returns_path_and_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "melody.wav"
    samples = np.zeros(10, dtype=np.float32)

    assert write_wav(samples, 22_050, path) == path

    pcm, sample_rate = read_wav(path)

    assert sample_rate == 22_050
    assert pcm.tolist() == [0] * 10


def test_header_is_mono_16_bit_pcm(tmp_path: Path) -> None:
    path = tmp_path / "melody.wav"
    write_wav(np.zeros(16, dtype=np.float32), 48_000, path)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 16


def test_samples_are_scaled_to_int16(tmp_path: Path) -> None:
    path = tmp_path / "melody.wav"
    samples = np.array([0.0, 0.6, -0.6, 0.25], dtype=np.float32)
    write_wav(samples, 22_050, path)

    pcm, _ = read_wav(path)

    assert pcm.tolist() == [0, 19660, -19660, 8192]


def test_out_of_range_samples_are_clipped(tmp_path: Path) -> None:
    path = tmp_path / "melody.wav"
    samples = np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float32)
    write_wav(samples, 22_050, path)

    pcm, _ = read_wav(path)

    assert pcm.tolist() == [32767, -32767, 32767, -32767]


def test_empty_audio_writes_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    write_wav(np.zeros(0, dtype=np.float32), 22_050, path)

    pcm, sample_rate = read_wav(path)

    assert sample_rate == 22_050
    assert pcm.size == 0


def test_multichannel_samples_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1-D"):
        write_wav(
            np.zeros((2, 10), dtype=np.float32),
            22_050,
            tmp_path / "melody.wav",
        )


def test_integer_samples_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="floating-point"):
        write_wav(
            np.zeros(10, dtype=np.int16),
            22_050,
            tmp_path / "melody.wav",
        )


def test_invalid_sample_rate_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        write_wav(
            np.zeros(10, dtype=np.float32),
            0,
            tmp_path / "melody.wav",
        )
