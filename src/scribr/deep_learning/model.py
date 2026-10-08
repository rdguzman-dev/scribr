"""Convolutional model for per-step pitch-sequence classification.

The input is a log-mel spectrogram and the output is one logit vector
per dataset step. The architecture is deliberately small: four
convolutional blocks over the time-frequency plane, a temporal
convolution, and a linear classifier. Attention, recurrence, and
frame-level onset detection are left to later experiments.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from ..representation import STEPS_PER_MELODY
from .features import DEFAULT_N_MELS
from .targets import NUM_CLASSES

DEFAULT_CHANNELS = (32, 64, 128, 256)
DEFAULT_DROPOUT = 0.1


class PitchSequenceCNN(nn.Module):
    """Map a log-mel spectrogram to one distribution over step classes.

    Args:
        n_mels: Number of mel channels in the input.
        channels: Output channels of each convolutional block. The last
            block keeps its time resolution and only pools frequency.
        steps: Number of dataset steps to predict.
        num_classes: Size of the per-step class vocabulary.
        dropout: Dropout probability before the classifier.

    Raises:
        ValueError: If `channels` is empty or a validation constraint
            is violated.
    """

    def __init__(
        self,
        *,
        n_mels: int = DEFAULT_N_MELS,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        steps: int = STEPS_PER_MELODY,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        if n_mels < 1:
            raise ValueError("n_mels must be at least 1")

        channels = tuple(int(channel) for channel in channels)

        if not channels:
            raise ValueError("channels must contain at least one block")

        if any(channel < 1 for channel in channels):
            raise ValueError("channels must be positive")

        if steps < 1:
            raise ValueError("steps must be at least 1")

        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        blocks: list[nn.Module] = []
        in_channels = 1

        for index, out_channels in enumerate(channels):
            # The last block keeps time resolution so the temporal
            # convolution and the step pooling still see fine detail.
            pool = (2, 1) if index == len(channels) - 1 else (2, 2)
            blocks.append(_conv_block(in_channels, out_channels, pool))
            in_channels = out_channels

        self.n_mels = int(n_mels)
        self.steps = int(steps)
        self.channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict per-step class logits.

        Args:
            features: `(batch, n_mels, time)` or
                `(batch, 1, n_mels, time)` log-mel features.

        Returns:
            A `(batch, steps, num_classes)` float tensor of logits.
        """
        if features.ndim == 3:
            features = features.unsqueeze(1)

        hidden = self.blocks(features)
        hidden = hidden.mean(dim=2)
        hidden = self.temporal(hidden)
        # Resample to the step grid. `adaptive_avg_pool1d` would be
        # equivalent, but MPS does not implement it for the
        # non-divisible sizes that arise here.
        hidden = F.interpolate(
            hidden,
            size=self.steps,
            mode="linear",
            align_corners=False,
        )

        return self.classifier(hidden.transpose(1, 2))


def _conv_block(
    in_channels: int,
    out_channels: int,
    pool: tuple[int, int],
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pool),
    )
