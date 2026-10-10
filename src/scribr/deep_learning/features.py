"""Log-mel spectrogram features for deep learning transcription.

Audio is always the mono `float32` output of `Synthesizer`, so the
extractor has no channel handling. Each example is standardized on its
own, which removes the loudness difference between instruments when a
model trained on one instrument is evaluated on another.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torchaudio
from torch import Tensor

DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 256
DEFAULT_N_MELS = 128
DEFAULT_F_MIN = 27.5
DEFAULT_F_MAX = 8_000.0
DEFAULT_TOP_DB = 80.0

# 8.55 seconds at the default rate and hop. Covers the 8-second melody
# plus the default 0.5-second release tail without cropping real audio.
DEFAULT_NUM_FRAMES = 736

# Floor for the standard deviation so silence does not get amplified.
_EPSILON = 1e-5


class FeatureExtractor(Protocol):
    """Audio feature contract shared by training and inference.

    Attributes:
        sample_rate: Sample rate of the expected input waveform in Hz.
        num_frames: Fixed length of the time axis, or `None` to keep
            the natural length.
    """

    sample_rate: int
    num_frames: int | None

    def spectrogram(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract features at their natural length."""
        ...

    def __call__(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract features cropped or padded to `num_frames`."""
        ...


class LogMelSpectrogram:
    """Extract standardized log-mel spectrograms from mono waveforms.

    Args:
        sample_rate: Sample rate of the input waveform in Hz.
        n_fft: FFT window size in samples.
        hop_length: FFT hop size in samples.
        n_mels: Number of mel filterbank channels.
        f_min: Lowest filter center frequency in Hz.
        f_max: Highest filter center frequency in Hz.
        top_db: Dynamic range clipped from the peak in dB.
        num_frames: Fixed length of the time axis. Shorter inputs are
            zero-padded, longer ones cropped. `None` keeps the natural
            length.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        f_min: float = DEFAULT_F_MIN,
        f_max: float = DEFAULT_F_MAX,
        top_db: float = DEFAULT_TOP_DB,
        num_frames: int | None = DEFAULT_NUM_FRAMES,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if n_fft < 1:
            raise ValueError("n_fft must be at least 1")

        if not 0 < hop_length <= n_fft:
            raise ValueError("hop_length must be in (0, n_fft]")

        if n_mels < 1:
            raise ValueError("n_mels must be at least 1")

        if f_min < 0:
            raise ValueError("f_min must be non-negative")

        if not f_min < f_max <= sample_rate / 2:
            raise ValueError(
                "f_max must be greater than f_min and at most the "
                "Nyquist frequency"
            )

        if top_db <= 0:
            raise ValueError("top_db must be positive")

        if num_frames is not None and num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.top_db = float(top_db)
        self.num_frames = None if num_frames is None else int(num_frames)

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=self.top_db,
        )

    def spectrogram(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract a standardized log-mel spectrogram at natural length.

        Empty audio has no spectrogram, so it becomes a zero feature
        map. The model then sees silence for an empty melody and can
        predict the all-`NOTE_OFF` sequence from context.

        Args:
            audio: Mono `float32` waveform.

        Returns:
            A `(n_mels, frames)` float tensor.
        """
        waveform = _as_waveform(audio)

        if waveform.numel() == 0:
            return torch.zeros(self.n_mels, 0, dtype=torch.float32)

        features = self._to_db(self._mel(waveform))

        return (features - features.mean()) / features.std().clamp_min(
            _EPSILON
        )

    def __call__(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract features cropped or padded to `num_frames`."""
        features = self.spectrogram(audio)

        if self.num_frames is None:
            return features

        return fix_length(features, self.num_frames)


def fix_length(features: Tensor, num_frames: int) -> Tensor:
    """Pad with zeros or crop so the time axis is exactly `num_frames`."""
    current = features.shape[-1]

    if current == num_frames:
        return features

    if current > num_frames:
        return features[..., :num_frames]

    padding = torch.zeros(
        *features.shape[:-1],
        num_frames - current,
        dtype=features.dtype,
    )

    return torch.cat([features, padding], dim=-1)


def _as_waveform(audio: np.ndarray | Tensor) -> Tensor:
    waveform = torch.as_tensor(audio, dtype=torch.float32)

    if waveform.ndim != 1:
        raise ValueError(
            f"audio must be one-dimensional, got shape "
            f"{tuple(waveform.shape)}"
        )

    return waveform
