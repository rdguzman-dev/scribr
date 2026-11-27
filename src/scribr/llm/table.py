"""Prompting and validation for the shared CSV-like melody format."""

import math

from scribr.representation import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    STEPS_PER_QUARTER_NOTE,
    Melody,
    Note,
    melody_from_text,
)

PROMPT_VERSION = 1
_HEADER = "pitch, onset, offset"


def transcription_prompt(tempo: float, duration_beats: float) -> str:
    """Request a note table without exposing reference notes to the model."""
    return f"""Transcribe the attached monophonic instrumental audio precisely.
Return only a CSV-like table with this header:
{_HEADER}
Each following row must contain a scientific pitch name (C4 is MIDI 60,
F#3 and Bb5 are valid), onset, and offset. Pitches must be in A0-C8
(MIDI {DATASET_MIN_PITCH}-{DATASET_MAX_PITCH}).
Times are quarter-note beats from the FIRST AUDIO SAMPLE, not seconds.
Tempo is {tempo:.12g} BPM; beats = seconds * {tempo:.12g} / 60.
Audio duration is {duration_beats:.12g} beats. All times must be finite,
0 <= onset < offset <= {duration_beats:.12g}.
Use decimal numbers and one row per note, ordered by onset, without overlap.
Preserve initial rests, gaps, and separate re-attacks of the same pitch.
Estimate actual note releases; do not transcribe reverberation or instrument
release tails as extra notes. Do not force timings onto a grid.
For silence, return ONLY the header. Do not include explanations, Markdown,
JSON, MIDI data, or extra columns."""


def parse_response(
    text: str,
    duration_beats: float,
    *,
    quantize: bool = True,
    time_origin_beats: float = 0.0,
) -> Melody:
    """Parse and validate a model table before optional 0.25-beat rounding.

    A header-only response denotes silence. An absent response is an error.
    A single Markdown fence is tolerated, but prose and invalid rows are not.

    Args:
        text: Completed model response in the shared note-table format.
        duration_beats: Maximum allowed note offset, including audio tails.
        quantize: Round times to the dataset's 16th-note grid.
        time_origin_beats: Leading silence removed from the supplied audio.
            Restore this origin before quantizing onto the original grid.

    Returns:
        A chronological, non-overlapping canonical `Melody`.

    Raises:
        ValueError: If the response is absent, malformed, out of range,
            unordered, overlapping, or extends beyond the audio.
    """
    if not math.isfinite(duration_beats) or duration_beats < 0:
        raise ValueError("duration_beats must be finite and non-negative")
    if not math.isfinite(time_origin_beats) or time_origin_beats < 0:
        raise ValueError("time_origin_beats must be finite and non-negative")

    table = text.strip()
    lines = table.splitlines()
    if lines and lines[0].strip().lower() in ("```", "```csv", "```text"):
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("incomplete table fence")
        table = "\n".join(lines[1:-1]).strip()

    if not table:
        raise ValueError("empty model response; silence requires the header")

    melody = melody_from_text(table)
    previous_offset = 0.0
    for note in melody:
        if not DATASET_MIN_PITCH <= note.pitch <= DATASET_MAX_PITCH:
            raise ValueError("pitch is outside the dataset range A0-C8")
        if note.onset < previous_offset:
            raise ValueError("notes must be ordered and non-overlapping")
        if note.offset > duration_beats + 1e-6:
            raise ValueError("note extends beyond the audio duration")
        previous_offset = note.offset

    if time_origin_beats:
        melody = Melody(
            tuple(
                Note(
                    note.pitch,
                    note.onset + time_origin_beats,
                    note.offset + time_origin_beats,
                )
                for note in melody
            )
        )

    if not quantize:
        return melody

    steps = STEPS_PER_QUARTER_NOTE
    last_step = math.floor((duration_beats + time_origin_beats + 1e-6) * steps)
    notes = []
    for note in melody:
        onset = math.floor(note.onset * steps + 0.5)
        offset = min(math.floor(note.offset * steps + 0.5), last_step)
        # Rounding can collapse a short note; never invent a longer note.
        if offset > onset:
            notes.append(Note(note.pitch, onset / steps, offset / steps))

    return Melody(tuple(notes))
