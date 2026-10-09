"""SoundFont resolution and shared synthesizer construction.

Materialization and training must render the same `SynthesisSpec` the
same way, so both build their synthesizers through `SynthesizerFactory`
instead of spelling out the `Synthesizer` arguments in two places.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from scribr.synthesis import SOUNDFONT_ENV_VAR, Instrument, Synthesizer

from .spec import SynthesisSpec


def resolve_soundfont_path(
    soundfont_path: str | Path | None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve `soundfont_path` or the `SCRIBR_SOUNDFONT` fallback.

    Args:
        soundfont_path: Explicit SoundFont path, or `None` to fall back
            to the `SCRIBR_SOUNDFONT` environment variable.
        required: Whether a missing SoundFont raises `ValueError`.

    Returns:
        The expanded SoundFont path, or `None` when `required` is
        `False` and no SoundFont is configured.

    Raises:
        ValueError: If `required` is `True` and no SoundFont is
            configured.
    """
    configured = (
        soundfont_path
        if soundfont_path is not None
        else os.environ.get(SOUNDFONT_ENV_VAR)
    )

    if not configured:
        if not required:
            return None

        raise ValueError(
            "No SoundFont configured. Pass soundfont_path=... or set the "
            f"{SOUNDFONT_ENV_VAR} environment variable."
        )

    return Path(configured).expanduser()


@dataclass(frozen=True, slots=True)
class SynthesizerFactory:
    """Picklable per-instrument synthesizer factory.

    Pickling matters because `DataLoader` workers receive the factory
    when audio is synthesized on the fly.

    Attributes:
        soundfont_path: Resolved SoundFont path shared by every
            instrument.
        synthesis: Rendering parameters shared by every instrument.
    """

    soundfont_path: Path
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
