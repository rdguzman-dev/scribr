"""Tests for parsing the baseline text note format into a `Melody`."""

from __future__ import annotations

import pytest

from scribr.representation.melody import Melody, Note
from scribr.representation.text import melody_from_text

SAMPLE = """\
pitch, onset, offset
C4, 0.00, 0.50
E4, 0.50, 1.00
G4, 1.00, 2.00
E4, 2.00, 2.50
"""


def test_parses_pitch_onset_offset_table() -> None:
    melody = melody_from_text(SAMPLE)

    assert melody == Melody(
        (
            Note(60, 0.0, 0.5),
            Note(64, 0.5, 1.0),
            Note(67, 1.0, 2.0),
            Note(64, 2.0, 2.5),
        )
    )


def test_exported_from_representation_package() -> None:
    from scribr.representation import melody_from_text as package_function

    assert package_function is melody_from_text


def test_header_is_optional_and_case_insensitive() -> None:
    without_header = melody_from_text("C4, 0.0, 0.5")
    mixed_case = melody_from_text("Pitch,Onset, OFFSET\nC4, 0.0, 0.5")

    assert without_header.notes == (Note(60, 0.0, 0.5),)
    assert mixed_case.notes == (Note(60, 0.0, 0.5),)


def test_blank_lines_and_surrounding_whitespace_are_ignored() -> None:
    text = "\n  pitch, onset, offset  \n\n   C4 , 0.25 , 0.75   \n\n"
    melody = melody_from_text(text)

    assert melody.notes == (Note(60, 0.25, 0.75),)


def test_empty_and_header_only_text_yield_empty_melody() -> None:
    assert melody_from_text("") == Melody()
    assert melody_from_text("pitch, onset, offset\n") == Melody()


def test_notes_keep_row_order() -> None:
    melody = melody_from_text("G4, 1.0, 2.0\nC4, 0.0, 1.0")

    assert melody.notes == (Note(67, 1.0, 2.0), Note(60, 0.0, 1.0))


def test_times_allow_decimal_beats() -> None:
    melody = melody_from_text("C4, 1.25, 3.5")

    assert melody.notes == (Note(60, 1.25, 3.5),)


@pytest.mark.parametrize(
    ("name", "pitch"),
    [
        ("C4", 60),
        ("c4", 60),
        ("A4", 69),
        ("C#4", 61),
        ("Db4", 61),
        ("E#4", 65),
        ("Fb4", 64),
        ("Bb3", 58),
        ("C-1", 0),
        ("A0", 21),
        ("G9", 127),
    ],
)
def test_pitch_names(name: str, pitch: int) -> None:
    melody = melody_from_text(f"{name}, 0.0, 1.0")

    assert melody.notes == (Note(pitch, 0.0, 1.0),)


def test_invalid_pitch_name_reports_line() -> None:
    with pytest.raises(ValueError, match=r"line 2: invalid pitch name 'H4'"):
        melody_from_text("C4, 0.0, 1.0\nH4, 1.0, 2.0")


@pytest.mark.parametrize("name", ["C##4", "Dbb4", "C#b4"])
def test_multiple_accidentals_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid pitch name"):
        melody_from_text(f"{name}, 0.0, 1.0")


def test_pitch_outside_midi_range_reports_line() -> None:
    with pytest.raises(ValueError, match=r"line 3: pitch 'C-2'"):
        melody_from_text("C4, 0.0, 1.0\nC4, 1.0, 2.0\nC-2, 2.0, 3.0")


def test_wrong_field_count_reports_line() -> None:
    with pytest.raises(
        ValueError, match=r"line 2: expected 3 comma-separated fields, got 2"
    ):
        melody_from_text("C4, 0.0, 1.0\nC4, 1.0")


def test_trailing_comma_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"line 1: expected 3 comma-separated fields"
    ):
        melody_from_text("C4, 0.0, 1.0,")


@pytest.mark.parametrize("label", ["onset", "offset"])
def test_non_numeric_time_reports_line(label: str) -> None:
    values = ["0.0", "not-a-number"]
    values[0 if label == "onset" else 1] = "not-a-number"

    with pytest.raises(
        ValueError, match=rf"line 1: {label} 'not-a-number' is not a number"
    ):
        melody_from_text(f"C4, {values[0]}, {values[1]}")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_time_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"line 1: onset must be finite"):
        melody_from_text(f"C4, {value}, 1.0")


def test_negative_onset_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"line 1: onset -0.5"):
        melody_from_text("C4, -0.5, 0.0")


def test_offset_must_be_greater_than_onset() -> None:
    with pytest.raises(
        ValueError, match=r"line 1: offset 0.5 must be greater than onset 0.5"
    ):
        melody_from_text("C4, 0.5, 0.5")

    with pytest.raises(ValueError, match=r"line 1: offset 0.25"):
        melody_from_text("C4, 0.5, 0.25")
