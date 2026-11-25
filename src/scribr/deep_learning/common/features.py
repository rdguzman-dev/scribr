"""Feature extraction contract shared by deep learning architectures."""

from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class FeatureExtractor(Protocol):
    """Audio feature contract shared by training and inference.

    A feature map may have one or more leading dimensions; its last axis
    is always time. Log-mel spectrograms use `(channels, frames)`, while
    Basic Pitch's harmonic CQT uses `(harmonics, bins, frames)`.

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
