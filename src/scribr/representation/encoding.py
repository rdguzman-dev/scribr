"""Conversion between the dataset's 64-step pitch sequences and a Scribr
`Melody`.

The 4 Bars Monophonic Melodies Dataset represents a melody as a sequence
of 64 integer dataset steps, with 4 steps per quarter note in 4/4 time.
Each step contains one of the following tokens:

- `21–108`: Start a note at that MIDI pitch.
- `128`: Continue the currently sounding note.
- `129`: End the current note, if any.

A new pitch token implicitly ends any note that is currently sounding.
"""

from collections.abc import Sequence

from .melody import (
    HOLD_TOKEN,
    NOTE_OFF_TOKEN,
    STEPS_PER_QUARTER_NOTE,
    Melody,
    Note,
)

# Allow tiny floating-point error when checking the dataset's step grid.
_QUANTIZATION_TOLERANCE = 1e-9


def decode_pitch_sequence(sequence: Sequence[int]) -> tuple[Note, ...]:
    """Decode a dataset pitch sequence into a tuple of `Note` objects.

    Args:
        sequence: Sequence of integer dataset tokens.

    Returns:
        A tuple of `Note` objects.
    """
    notes: list[Note] = []
    active_pitch: int | None = None
    active_start = 0

    def close_active(end_step: int) -> None:
        nonlocal active_pitch

        if active_pitch is not None:
            notes.append(
                Note(
                    active_pitch,
                    _step_to_beat(active_start),
                    _step_to_beat(end_step),
                )
            )
            active_pitch = None

    for step, token in enumerate(sequence):
        if token == HOLD_TOKEN:
            continue

        close_active(step)

        if token != NOTE_OFF_TOKEN:
            active_pitch = token
            active_start = step

    close_active(len(sequence))

    return tuple(notes)


def encode_pitch_sequence(
    melody: Melody,
    *,
    num_steps: int | None = None,
) -> tuple[int, ...]:
    """Encode a Scribr `Melody` back into a dataset pitch sequence.

    This should mainly be used for round-trip tests. Transcription
    approaches should produce `Melody` objects directly instead.

    Args:
        melody: The melody to encode.
        num_steps: Length of the returned sequence. Defaults to the
            number of steps needed to cover the last note.

    Returns:
        A tuple of integer pitch tokens representing the melody as a
        dataset pitch sequence.

    Raises:
        ValueError: If a note's onset or offset is not on the dataset's
            step grid, notes overlap, or the requested `num_steps` is
            too short.
    """
    # (onset_step, offset_step, pitch)
    events: list[tuple[int, int, int]] = []
    previous_offset = 0

    for note in sorted(melody.notes, key=lambda item: item.onset):
        onset = _beat_to_step(note.onset, label="onset")
        offset = _beat_to_step(note.offset, label="offset")

        if offset <= onset:
            raise ValueError(
                f"note {note.pitch} has zero length on the dataset's step grid"
            )

        if onset < previous_offset:
            raise ValueError(
                "overlapping notes cannot be encoded monophonically"
            )

        previous_offset = offset
        events.append((onset, offset, note.pitch))

    required = events[-1][1] if events else 0  # Last event's offset

    if num_steps is None:
        num_steps = required

    elif num_steps < required:
        raise ValueError(
            f"num_steps={num_steps} is too short for a melody ending at "
            f"step {required}"
        )

    sequence = [NOTE_OFF_TOKEN] * num_steps

    for onset, offset, pitch in events:
        sequence[onset] = pitch

        for step in range(onset + 1, offset):
            sequence[step] = HOLD_TOKEN

    return tuple(sequence)


def _step_to_beat(step: int) -> float:
    """Convert a dataset step index to a quarter-note beat position."""
    return step / STEPS_PER_QUARTER_NOTE


def _beat_to_step(beat: float, *, label: str) -> int:
    """Convert a quarter-note beat position to a dataset step index."""
    step = beat * STEPS_PER_QUARTER_NOTE
    rounded = round(step)

    if abs(step - rounded) > _QUANTIZATION_TOLERANCE:
        raise ValueError(
            f"{label} {beat} beats is not on the dataset's "
            "4-steps-per-quarter-note grid"
        )

    if rounded < 0:
        raise ValueError(f"{label} {beat} beats must be non-negative")

    return rounded
