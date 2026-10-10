"""Dataset loading and synthesizer construction shared by experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from experiments.common.synthesis import (
    SynthesizerFactory,
    resolve_soundfont_path,
)
from scribr.data import MelodyDataset
from scribr.representation import MelodyExample

from .spec import DataSpec, TrainingConfig


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


def default_synthesizer_factory(
    config: TrainingConfig,
    soundfont_path: str | Path | None,
) -> SynthesizerFactory:
    """Build the factory, resolving the SoundFont from the
    environment."""
    return SynthesizerFactory(
        soundfont_path=resolve_soundfont_path(soundfont_path),
        synthesis=config.synthesis,
    )
