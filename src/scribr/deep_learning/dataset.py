"""Synthesis-on-demand dataset for pitch-sequence training.

`SynthesizedMelodyDataset` wraps decoded `MelodyExample` objects and
turns each one into a `(features, target)` pair on access. Synthesis is
the expensive step, so the dataset is map-style: a `DataLoader` can
shuffle it reproducibly and fan it out over workers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..representation import MelodyExample
from ..synthesis import Synthesizer
from .features import FeatureExtractor
from .targets import pitch_sequence_to_classes


class SynthesizedMelodyDataset(Dataset):
    """Yield `(log-mel features, target classes)` pairs.

    Args:
        examples: Decoded melodies to train or evaluate on.
        synthesizer_factory: Zero-argument callable returning a
            `Synthesizer`. It is called once per instance, so each
            `DataLoader` worker builds its own synthesizer. A
            `Synthesizer` holds a module handle and cannot be pickled,
            which is why the factory exists instead of a synthesizer.
        features: Feature extractor applied to each synthesized
            waveform. Its sample rate must match the synthesizer's.

    Note:
        The target is the raw `MelodyExample.pitch_sequence` mapped to
        class indices, so a melody whose record is shorter than the
        model's step count would produce a shorter target and fail in
        the loss. Dataset records are always `STEPS_PER_MELODY` long.
    """

    def __init__(
        self,
        examples: Sequence[MelodyExample],
        synthesizer_factory: Callable[[], Synthesizer],
        features: FeatureExtractor,
    ) -> None:
        self.examples = tuple(examples)
        self.synthesizer_factory = synthesizer_factory
        self.features = features
        self._synthesizer: Synthesizer | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        example = self.examples[index]
        audio = self._synthesizer_instance().synthesize(example.melody)

        return (
            self.features(audio),
            torch.tensor(
                pitch_sequence_to_classes(example.pitch_sequence),
                dtype=torch.long,
            ),
        )

    def _synthesizer_instance(self) -> Synthesizer:
        if self._synthesizer is None:
            synthesizer = self.synthesizer_factory()

            if synthesizer.sample_rate != self.features.sample_rate:
                raise ValueError(
                    f"synthesizer sample rate {synthesizer.sample_rate} "
                    f"does not match feature sample rate "
                    f"{self.features.sample_rate}"
                )

            self._synthesizer = synthesizer

        return self._synthesizer
