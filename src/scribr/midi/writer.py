"""Optional MIDI export for Scribr melodies and transcriptions.

MIDI is an *interchange/export* format only. Dataset loading, training,
evaluation and synthesis never require it. This module exists so a
finished transcription can be opened in a DAW or shared with other
tools.

Tempo and MIDI program are configurable, and the default MIDI resolution
uses 120 ticks per dataset step.
"""

from dataclasses import dataclass
from pathlib import Path

import mido

from ..representation.melody import STEPS_PER_QUARTER_NOTE, Melody

DEFAULT_TEMPO_BPM = 120.0
DEFAULT_PROGRAM = 0
DEFAULT_CHANNEL = 0
DEFAULT_VELOCITY = 80
# 120 MIDI ticks per dataset step.
DEFAULT_TICKS_PER_QUARTER_NOTE = 120 * STEPS_PER_QUARTER_NOTE


@dataclass(frozen=True, slots=True)
class MidiExportConfig:
    """Configuration for `melody_to_midi` and `write_midi`.

    Attributes:
        tempo: Tempo (in BPM) written as a `set_tempo` meta message.
        program: General MIDI program number (`0` = Acoustic Grand
            Piano).
        channel: MIDI channel in the range `[0, 15]`.
        velocity: Note-on velocity in the range `[1, 127]`.
        ticks_per_beat: MIDI timing resolution in ticks per quarter-note
            beat.

    Raises:
        ValueError: If `tempo` is not positive, `program`, `channel`, or
            `velocity` is outside its MIDI range, or `ticks_per_beat` is
            less than `1`.
    """

    tempo: float = DEFAULT_TEMPO_BPM
    program: int = DEFAULT_PROGRAM
    channel: int = DEFAULT_CHANNEL
    velocity: int = DEFAULT_VELOCITY
    ticks_per_beat: int = DEFAULT_TICKS_PER_QUARTER_NOTE

    def __post_init__(self) -> None:
        if self.tempo <= 0:
            raise ValueError("tempo must be positive")

        if not 0 <= self.program <= 127:
            raise ValueError("program must be in [0, 127]")

        if not 0 <= self.channel <= 15:
            raise ValueError("channel must be in [0, 15]")

        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be in [1, 127]")

        if self.ticks_per_beat < 1:
            raise ValueError("ticks_per_beat must be at least 1")


def melody_to_midi(
    melody: Melody,
    config: MidiExportConfig | None = None,
) -> mido.MidiFile:
    """Convert a `Melody` to an in-memory `mido.MidiFile`.

    The result is a single track with a single melodic line. At the
    same tick, note-off events are ordered before note-on events so
    that adjacent notes with the same pitch (a common pattern in the
    dataset) are represented as separate notes rather than being merged.

    Args:
        melody: The melody to convert.
        config: Export configuration. Defaults to `MidiExportConfig()`.

    Returns:
        An in-memory `mido.MidiFile` containing one track.
    """
    if config is None:
        config = MidiExportConfig()

    midi = mido.MidiFile(ticks_per_beat=config.ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    def beat_to_tick(beat: float) -> int:
        return int(round(beat * config.ticks_per_beat))

    # (absolute_tick, sort_priority, message); lower priority sorts first.
    events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = [
        (
            0,
            0,
            mido.MetaMessage(
                "set_tempo", tempo=mido.bpm2tempo(config.tempo), time=0
            ),
        ),
        (
            0,
            1,
            mido.Message(
                "program_change",
                program=config.program,
                channel=config.channel,
                time=0,
            ),
        ),
    ]

    for note in melody.notes:
        events.append(
            (
                beat_to_tick(note.onset),
                3,
                mido.Message(
                    "note_on",
                    note=note.pitch,
                    velocity=config.velocity,
                    channel=config.channel,
                    time=0,
                ),
            )
        )
        events.append(
            (
                beat_to_tick(note.offset),
                2,
                mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    channel=config.channel,
                    time=0,
                ),
            )
        )

    events.sort(key=lambda event: (event[0], event[1]))
    previous_tick = 0

    for absolute_tick, _, message in events:
        message.time = absolute_tick - previous_tick
        previous_tick = absolute_tick
        track.append(message)

    track.append(mido.MetaMessage("end_of_track", time=0))

    return midi


def write_midi(
    melody: Melody,
    path: str | Path,
    config: MidiExportConfig | None = None,
) -> Path:
    """Write a `Melody` to `path` as a standard MIDI file.

    Args:
        melody: The melody to write.
        path: Destination path for the MIDI file.
        config: Export configuration. Defaults to `MidiExportConfig()`.

    Returns:
        The path to the written MIDI file.
    """
    output = Path(path)
    melody_to_midi(melody, config).save(output)

    return output
