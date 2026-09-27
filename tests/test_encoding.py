"""Tests for pitch-sequence <-> Melody conversion semantics."""

from __future__ import annotations

import pytest

from scribr.representation.encoding import (
    decode_pitch_sequence,
    encode_pitch_sequence,
)
from scribr.representation.melody import Melody, Note


def test_decode_basic_note_hold_and_rest() -> None:
    notes = decode_pitch_sequence([60, 128, 129, 62])
    assert notes == (Note(60, 0.0, 0.5), Note(62, 0.75, 1.0))


def test_decode_leading_and_consecutive_note_offs_are_rests() -> None:
    notes = decode_pitch_sequence([129, 129, 129, 60, 128])
    assert notes == (Note(60, 0.75, 1.25),)


def test_decode_note_through_final_step() -> None:
    notes = decode_pitch_sequence([129, 60, 128])
    assert notes == (Note(60, 0.25, 0.75),)


def test_decode_consecutive_identical_pitches_are_separate_notes() -> None:
    notes = decode_pitch_sequence([60, 60])
    assert notes == (Note(60, 0.0, 0.25), Note(60, 0.25, 0.5))


def test_decode_new_pitch_implicitly_ends_previous_note() -> None:
    notes = decode_pitch_sequence([60, 64, 129])
    assert notes == (Note(60, 0.0, 0.25), Note(64, 0.25, 0.5))


def test_decode_hold_without_active_note_is_silence() -> None:
    notes = decode_pitch_sequence([128, 60, 128])
    assert notes == (Note(60, 0.25, 0.75),)


def test_decode_empty_sequence() -> None:
    assert decode_pitch_sequence([]) == ()


def test_decode_timings_are_exact_multiples_of_quarter_step() -> None:
    notes = decode_pitch_sequence([60, 128, 128, 62, 129, 128])
    for note in notes:
        assert note.onset * 4 == round(note.onset * 4)
        assert note.offset * 4 == round(note.offset * 4)


@pytest.mark.parametrize(
    "sequence",
    [
        [63, 128, 128, 64, 128, 128, 129, 57],
        [129, 129, 60, 128, 129, 62, 62, 129],
        [60] * 8,
        [129] * 8,
    ],
)
def test_encode_is_inverse_of_decode(sequence: list[int]) -> None:
    notes = decode_pitch_sequence(sequence)
    assert encode_pitch_sequence(
        Melody(notes), num_steps=len(sequence)
    ) == tuple(sequence)


def test_encode_defaults_to_length_of_last_note() -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(62, 1.0, 1.25)))
    assert encode_pitch_sequence(melody) == (60, 128, 129, 129, 62)


def test_encode_empty_melody() -> None:
    assert encode_pitch_sequence(Melody()) == ()


def test_encode_rejects_off_grid_timing() -> None:
    with pytest.raises(ValueError, match="grid"):
        encode_pitch_sequence(Melody((Note(60, 0.0, 0.3),)))


def test_encode_rejects_overlapping_notes() -> None:
    melody = Melody((Note(60, 0.0, 1.0), Note(62, 0.5, 1.5)))
    with pytest.raises(ValueError, match="overlapping"):
        encode_pitch_sequence(melody)


def test_encode_rejects_too_short_requested_length() -> None:
    melody = Melody((Note(60, 0.0, 1.0),))
    with pytest.raises(ValueError, match="too short"):
        encode_pitch_sequence(melody, num_steps=2)


def test_note_validation() -> None:
    with pytest.raises(ValueError):
        Note(200, 0.0, 1.0)
    with pytest.raises(ValueError):
        Note(60, 1.0, 1.0)
    with pytest.raises(ValueError):
        Note(60, -0.5, 1.0)


def test_melody_accepts_any_iterable_of_notes() -> None:
    melody = Melody([Note(60, 0.0, 0.5)])

    assert isinstance(melody.notes, tuple)
    assert list(melody) == [Note(60, 0.0, 0.5)]
    assert melody.duration == 0.5
