"""Tests for the General MIDI instrument vocabulary."""

from __future__ import annotations

import json

import pytest

from scribr.synthesis import INSTRUMENT_PROGRAMS, Instrument


def test_members_serialize_to_plain_names() -> None:
    assert Instrument.PIANO == "piano"
    assert json.dumps(Instrument.ACOUSTIC_GUITAR) == '"acoustic_guitar"'
    assert [instrument.value for instrument in Instrument] == [
        "piano",
        "acoustic_guitar",
        "electric_guitar",
        "violin",
    ]


def test_general_midi_programs() -> None:
    assert INSTRUMENT_PROGRAMS == {
        Instrument.PIANO: 0,
        Instrument.ACOUSTIC_GUITAR: 25,
        Instrument.ELECTRIC_GUITAR: 27,
        Instrument.VIOLIN: 40,
    }
    assert Instrument.PIANO.program == 0
    assert Instrument.ACOUSTIC_GUITAR.program == 25
    assert Instrument.ELECTRIC_GUITAR.program == 27
    assert Instrument.VIOLIN.program == 40


def test_string_round_trip() -> None:
    assert Instrument("violin") is Instrument.VIOLIN
    assert Instrument(Instrument.VIOLIN.value) is Instrument.VIOLIN


def test_unknown_instrument_raises() -> None:
    with pytest.raises(ValueError):
        Instrument("cello")
