"""Tests for optional MIDI export, inspecting resulting events."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from scribr.midi.writer import MidiExportConfig, melody_to_midi, write_midi
from scribr.representation.melody import Melody, Note


def absolute_events(midi: mido.MidiFile) -> list[tuple[int, mido.Message]]:
    """Flatten the first track to absolute-tick events."""
    events = []
    tick = 0
    for message in midi.tracks[0]:
        tick += message.time
        events.append((tick, message))
    return events


def note_events(midi: mido.MidiFile) -> list[tuple[int, str, int]]:
    return [
        (tick, message.type, message.note)
        for tick, message in absolute_events(midi)
        if message.type in ("note_on", "note_off")
    ]


def test_single_monophonic_track() -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(64, 1.0, 1.5)))
    midi = melody_to_midi(melody)
    assert len(midi.tracks) == 1


def test_note_on_off_ticks() -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(64, 1.0, 1.5)))
    midi = melody_to_midi(melody)
    assert note_events(midi) == [
        (0, "note_on", 60),
        (240, "note_off", 60),
        (480, "note_on", 64),
        (720, "note_off", 64),
    ]


def test_hold_spans_multiple_steps() -> None:
    melody = Melody((Note(60, 0.0, 0.75),))
    assert note_events(melody_to_midi(melody)) == [
        (0, "note_on", 60),
        (360, "note_off", 60),
    ]


def test_rest_leaves_a_gap() -> None:
    melody = Melody((Note(60, 0.0, 0.25), Note(62, 1.0, 1.25)))
    assert note_events(melody_to_midi(melody)) == [
        (0, "note_on", 60),
        (120, "note_off", 60),
        (480, "note_on", 62),
        (600, "note_off", 62),
    ]


def test_adjacent_same_pitch_gets_note_off_before_note_on() -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(60, 0.5, 1.0)))
    events = note_events(melody_to_midi(melody))
    assert events == [
        (0, "note_on", 60),
        (240, "note_off", 60),
        (240, "note_on", 60),
        (480, "note_off", 60),
    ]


def test_note_continuing_through_final_step_is_terminated() -> None:
    melody = Melody((Note(60, 15.75, 16.0),))
    events = note_events(melody_to_midi(melody))
    assert events == [(120 * 63, "note_on", 60), (120 * 64, "note_off", 60)]


def test_quantization_matches_dataset_grid() -> None:
    # 4 steps per quarter -> 120 ticks per step at 480 ticks/beat.
    melody = Melody((Note(60, 0.25, 0.5),))
    assert note_events(melody_to_midi(melody)) == [
        (120, "note_on", 60),
        (240, "note_off", 60),
    ]


def test_tempo_and_program_are_configurable() -> None:
    config = MidiExportConfig(tempo=90.0, program=42, channel=3, velocity=100)
    midi = melody_to_midi(Melody((Note(60, 0.0, 0.25),)), config)

    events = absolute_events(midi)
    tempo = next(message for _, message in events if message.type == "set_tempo")
    program = next(
        message for _, message in events if message.type == "program_change"
    )
    note_on = next(message for _, message in events if message.type == "note_on")

    assert tempo.tempo == mido.bpm2tempo(90.0)
    assert program.program == 42
    assert program.channel == 3
    assert note_on.velocity == 100
    assert note_on.channel == 3


def test_default_config_is_120_bpm_piano() -> None:
    midi = melody_to_midi(Melody((Note(60, 0.0, 0.25),)))
    events = absolute_events(midi)
    tempo = next(message for _, message in events if message.type == "set_tempo")
    program = next(
        message for _, message in events if message.type == "program_change"
    )
    assert tempo.tempo == mido.bpm2tempo(120.0)
    assert program.program == 0


def test_track_ends_at_last_note_off() -> None:
    midi = melody_to_midi(Melody((Note(60, 0.0, 1.0),)))
    events = absolute_events(midi)
    assert events[-1][1].type == "end_of_track"
    assert events[-1][0] == 480


def test_empty_melody_produces_valid_file() -> None:
    midi = melody_to_midi(Melody())
    types = [message.type for _, message in absolute_events(midi)]
    assert "end_of_track" in types
    assert "note_on" not in types


def test_write_midi_round_trip(tmp_path: Path) -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(60, 0.5, 1.0), Note(64, 2.0, 2.75)))
    path = tmp_path / "melody.mid"
    returned = write_midi(melody, path, MidiExportConfig(tempo=100.0))

    assert returned == path
    assert path.exists()

    loaded = mido.MidiFile(path)
    expected = melody_to_midi(melody, MidiExportConfig(tempo=100.0))
    assert note_events(loaded) == note_events(expected)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        MidiExportConfig(program=128)
    with pytest.raises(ValueError):
        MidiExportConfig(velocity=0)
    with pytest.raises(ValueError):
        MidiExportConfig(channel=16)
    with pytest.raises(ValueError):
        MidiExportConfig(tempo=0)
    with pytest.raises(ValueError):
        MidiExportConfig(ticks_per_beat=0)
