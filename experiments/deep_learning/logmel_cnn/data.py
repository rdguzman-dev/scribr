"""Data, feature, and model construction for the log-mel CNN
experiment."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from experiments.common.spec import SynthesisSpec
from scribr.data import MelodyDataset
from scribr.deep_learning import LogMelSpectrogram, PitchSequenceCNN
from scribr.representation import STEPS_PER_MELODY, MelodyExample
from scribr.synthesis import SOUNDFONT_ENV_VAR, Instrument, Synthesizer

from .spec import DataSpec, FeaturesSpec, ModelSpec, TrainingConfig


@dataclass(frozen=True, slots=True)
class SynthesizerFactory:
    """Picklable per-instrument synthesizer factory."""

    soundfont_path: str
    synthesis: SynthesisSpec

    def __call__(self, instrument: Instrument) -> Synthesizer:
        return Synthesizer(
            self.soundfont_path,
            sample_rate=self.synthesis.sample_rate,
            tempo=self.synthesis.tempo,
            program=instrument.program,
            velocity=self.synthesis.velocity,
            gain=self.synthesis.gain,
            release_tail_seconds=self.synthesis.release_tail_seconds,
        )


def load_examples(
    spec: DataSpec,
    *,
    root: str | Path | None = None,
) -> list[MelodyExample]:
    """Load the first `spec.max_examples` records of a split.

    The subset is a prefix of the stored shard order, so it is
    deterministic and independent of `DataLoader` worker count. Records
    are held as decoded examples; audio is still synthesized on demand.
    """
    dataset = MelodyDataset(split=spec.split, root=root, shuffle=False)
    examples = list(islice(dataset, spec.max_examples))

    if not examples:
        raise ValueError(f"{spec.split!r} split contains no examples")

    return examples


def subset_fingerprint(examples: Iterable[MelodyExample]) -> str:
    """Return a SHA-256 digest of the pitch sequences in a subset."""
    digest = hashlib.sha256()

    for example in examples:
        # Dataset tokens are 21-129, so each one fits in one byte.
        digest.update(bytes(example.pitch_sequence))

    return digest.hexdigest()


def build_features(spec: FeaturesSpec) -> LogMelSpectrogram:
    """Build the feature extractor described by `spec`."""
    return LogMelSpectrogram(**spec.to_dict())


def build_model(spec: ModelSpec, *, n_mels: int) -> PitchSequenceCNN:
    """Build the model described by `spec` for `n_mels` input bands."""
    return PitchSequenceCNN(
        n_mels=n_mels,
        channels=spec.channels,
        steps=STEPS_PER_MELODY,
        dropout=spec.dropout,
    )


def default_synthesizer_factory(
    config: TrainingConfig,
    soundfont_path: str | Path | None,
) -> SynthesizerFactory:
    """Build the factory, resolving the SoundFont from the
    environment."""
    configured = (
        soundfont_path
        if soundfont_path is not None
        else os.environ.get(SOUNDFONT_ENV_VAR)
    )

    if configured is None:
        raise ValueError(
            "No SoundFont configured. Pass soundfont_path=... or set the "
            f"{SOUNDFONT_ENV_VAR} environment variable."
        )

    return SynthesizerFactory(
        soundfont_path=str(Path(configured).expanduser()),
        synthesis=config.synthesis,
    )
