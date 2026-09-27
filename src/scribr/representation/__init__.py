"""Canonical symbolic representation for Scribr."""

from .encoding import (
    decode_pitch_sequence,
    encode_pitch_sequence,
)
from .melody import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    HOLD_TOKEN,
    MIDI_MAX_PITCH,
    MIDI_MIN_PITCH,
    NOTE_OFF_TOKEN,
    STEPS_PER_BAR,
    STEPS_PER_MELODY,
    STEPS_PER_QUARTER_NOTE,
    Melody,
    MelodyExample,
    Note,
)
from .text import melody_from_text, melody_to_text

__all__ = [
    "DATASET_MAX_PITCH",
    "DATASET_MIN_PITCH",
    "HOLD_TOKEN",
    "MIDI_MAX_PITCH",
    "MIDI_MIN_PITCH",
    "NOTE_OFF_TOKEN",
    "STEPS_PER_BAR",
    "STEPS_PER_MELODY",
    "STEPS_PER_QUARTER_NOTE",
    "Melody",
    "MelodyExample",
    "Note",
    "decode_pitch_sequence",
    "encode_pitch_sequence",
    "melody_from_text",
    "melody_to_text",
]
