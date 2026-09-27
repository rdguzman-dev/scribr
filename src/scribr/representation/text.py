"""Conversion between CSV-like note-table text and a Scribr `Melody`.

This is the intermediary format for the LLM and human baseline
approaches. A melody is written as a CSV-like table::

    pitch, onset, offset
    C4, 0.00, 0.50
    E4, 0.50, 1.00

Pitches use scientific pitch notation (`C4` is MIDI 60) with at most one
`#` or `b` accidental. Onset and offset are quarter-note beats from the
start of the melody, matching `Note`. The header row is optional, blank
lines are ignored, and notes keep the order of the rows.
"""

import math
import re

from .melody import MIDI_MAX_PITCH, MIDI_MIN_PITCH, Melody, Note

_HEADER_FIELDS = ("pitch", "onset", "offset")

# Scientific pitch name: letter, accidental, octave (C4, F#3, Bb5).
_PITCH_NAME = re.compile(r"([A-Ga-g])([#b]?)(-?\d+)")

# Semitone offset of each natural pitch class relative to C.
_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_ACCIDENTAL_OFFSETS = {"#": 1, "b": -1}

# Pitch class names in sharp-only spelling, indexed by semitone.
_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


def melody_to_text(melody: Melody) -> str:
    """Serialize a `Melody` to a `pitch, onset, offset` table.

    The output has no header, one note per row, and a trailing newline
    after the last note. Pitch names always use sharps and times are
    written with `str(float)` so they round-trip exactly through
    `melody_from_text`.

    Args:
        melody: Melody to serialize.

    Returns:
        The note table, or `""` for an empty melody.
    """
    lines = [
        f"{_midi_to_pitch_name(note.pitch)}, {note.onset}, {note.offset}"
        for note in melody.notes
    ]

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def _midi_to_pitch_name(pitch: int) -> str:
    return f"{_PITCH_CLASS_NAMES[pitch % 12]}{pitch // 12 - 1}"


def melody_from_text(text: str) -> Melody:
    """Parse a `pitch, onset, offset` table into a `Melody`.

    Pitches are scientific pitch notation (`C4` is MIDI 60) with at
    most one `#` or `b` accidental. Times are quarter-note beats. The
    parser does not sort or quantize the notes; they keep the order of
    the rows.

    Args:
        text: Text containing one note per row.

    Returns:
        The parsed melody, or an empty melody when there are no note
        rows.

    Raises:
        ValueError: If a row does not have three fields, a pitch name is
            invalid or outside the MIDI range, a time is not a finite
            number, or a note's offset is not greater than its onset.
    """
    notes: list[Note] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        fields = [field.strip() for field in line.split(",")]

        if tuple(field.lower() for field in fields) == _HEADER_FIELDS:
            continue

        if len(fields) != 3:
            raise ValueError(
                f"line {line_number}: expected 3 comma-separated fields, "
                f"got {len(fields)}: {line!r}"
            )

        pitch_field, onset_field, offset_field = fields

        try:
            pitch = _pitch_name_to_midi(pitch_field)
            onset = _parse_time(onset_field, "onset")
            offset = _parse_time(offset_field, "offset")
            note = Note(pitch, onset, offset)

        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error

        notes.append(note)

    return Melody(notes)


def _pitch_name_to_midi(name: str) -> int:
    match = _PITCH_NAME.fullmatch(name)

    if match is None:
        raise ValueError(
            f"invalid pitch name {name!r}; expected scientific pitch "
            "notation such as C4, F#3, or Bb5"
        )

    letter, accidental, octave = match.groups()
    pitch = 12 * (int(octave) + 1) + _PITCH_CLASSES[letter.upper()]

    if accidental:
        pitch += _ACCIDENTAL_OFFSETS[accidental]

    if not MIDI_MIN_PITCH <= pitch <= MIDI_MAX_PITCH:
        raise ValueError(
            f"pitch {name!r} is outside the MIDI range "
            f"[{MIDI_MIN_PITCH}, {MIDI_MAX_PITCH}]"
        )

    return pitch


def _parse_time(value: str, label: str) -> float:
    try:
        time = float(value)

    except ValueError as error:
        raise ValueError(f"{label} {value!r} is not a number") from error

    if not math.isfinite(time):
        raise ValueError(f"{label} must be finite")

    return time
