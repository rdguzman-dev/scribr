"""On-demand audio synthesis of Scribr melodies."""

from .instruments import INSTRUMENT_PROGRAMS, Instrument
from .synthesizer import (
    DEFAULT_GAIN,
    DEFAULT_PROGRAM,
    DEFAULT_RELEASE_TAIL_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEMPO_BPM,
    DEFAULT_VELOCITY,
    SOUNDFONT_ENV_VAR,
    Synthesizer,
)

__all__ = [
    "DEFAULT_GAIN",
    "DEFAULT_PROGRAM",
    "DEFAULT_RELEASE_TAIL_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TEMPO_BPM",
    "DEFAULT_VELOCITY",
    "INSTRUMENT_PROGRAMS",
    "SOUNDFONT_ENV_VAR",
    "Instrument",
    "Synthesizer",
]
