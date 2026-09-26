"""Canonical in-memory symbolic representation of a melody.

Everything downstream of the dataset (synthesis, transcription
approaches, evaluation, MIDI export) speaks in terms of `Note`,
`Melody`, and `MelodyExample`.

Timing is expressed as quarter-note beats measured from the start of the
melody, *not* seconds and *not* dataset steps. The 4 Bars Monophonic
Melodies dataset uses 4/4 time and is quantized to
`STEPS_PER_QUARTER_NOTE = 4` steps per quarter note, so every
onset/offset decoded from it is an exact multiple of `0.25` quarter-note
beats.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

# Dataset timing: 4 steps per quarter note in 4/4 time.
STEPS_PER_QUARTER_NOTE = 4
STEPS_PER_BAR = 4 * STEPS_PER_QUARTER_NOTE
STEPS_PER_MELODY = 4 * STEPS_PER_BAR  # Dataset melodies are 4 bars long.

HOLD_TOKEN = 128  # "continue the currently sounding note".
NOTE_OFF_TOKEN = 129  # "no note is sounding".

# Dataset pitch range in MIDI note numbers.
DATASET_MIN_PITCH = 21
DATASET_MAX_PITCH = 108

# Full MIDI pitch range.
MIDI_MIN_PITCH = 0
MIDI_MAX_PITCH = 127


@dataclass(frozen=True, slots=True)
class Note:
    """A single monophonic note.

    Attributes:
        pitch: MIDI pitch number in the range `[0, 127]`.
        onset: Start time in quarter-note beats from the melody start.
        offset: End time in quarter-note beats from the melody start.

    Raises:
        TypeError: If `pitch` is not an integer.
        ValueError: If `pitch` is outside the MIDI range, `onset` is
            negative, or `onset` is greater than `offset`.
    """

    pitch: int
    onset: float
    offset: float

    def __post_init__(self) -> None:
        if not isinstance(self.pitch, int):
            raise TypeError(
                f"pitch must be an int, got {type(self.pitch).__name__}"
            )

        if not MIDI_MIN_PITCH <= self.pitch <= MIDI_MAX_PITCH:
            raise ValueError(
                f"pitch {self.pitch} is outside the MIDI range "
                f"[{MIDI_MIN_PITCH}, {MIDI_MAX_PITCH}]"
            )

        onset = float(self.onset)
        offset = float(self.offset)

        if onset < 0.0:
            raise ValueError(f"onset {onset} must be non-negative")

        if offset <= onset:
            raise ValueError(
                f"offset {offset} must be greater than onset {onset}"
            )

        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "offset", offset)

    @property
    def duration(self) -> float:
        """Note duration in quarter-note beats."""
        return self.offset - self.onset


@dataclass(frozen=True, slots=True)
class Melody:
    """A symbolic representation of a monophonic melody.

    This is the shared output type for every transcription approach
    (algorithmic, deep learning, LLM). Notes are expected to be in onset
    order and non-overlapping, but that is not enforced here so that raw
    predictions can be carried around and validated by evaluators.

    Attributes:
        notes: Notes comprising the melody.
    """

    notes: tuple[Note, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))

    def __iter__(self):
        return iter(self.notes)

    def __len__(self) -> int:
        return len(self.notes)

    @property
    def duration(self) -> float:
        """End time of the melody in quarter-note beats (0.0 if
        empty)."""
        return max((note.offset for note in self.notes), default=0.0)


@dataclass(frozen=True, slots=True)
class MelodyExample:
    """A decoded dataset melody together with its original
    representation and metadata.

    Attributes:
        melody: Decoded symbolic representation of the melody.
        pitch_sequence: Raw 64-step token sequence from the TFRecord.
        attributes: Original 13 context attributes computed by the
            dataset authors, stored as a plain mapping so that the
            object is easy to pickle. Treat this as read-only.
    """

    melody: Melody
    pitch_sequence: tuple[int, ...] = ()
    attributes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pitch_sequence", tuple(self.pitch_sequence))
        object.__setattr__(self, "attributes", dict(self.attributes))
