"""Tests for on-demand FluidSynth synthesis.

Unit tests use a fake `fluidsynth` module so they run anywhere.  The
integration test at the end is skipped unless `SCRIBR_SOUNDFONT` points
at a real SoundFont and the system FluidSynth library is installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scribr.data.dataset import MelodyDataset
from scribr.representation.melody import Melody, Note
from scribr.synthesis.synthesizer import SOUNDFONT_ENV_VAR, Synthesizer


class FakeSynth:
    instances: list["FakeSynth"] = []

    def __init__(self, *, gain: float, samplerate: float) -> None:
        self.gain = gain
        self.samplerate = samplerate
        self.calls: list[tuple] = []
        self.deleted = False
        FakeSynth.instances.append(self)

    def sfload(self, path: str) -> int:
        self.calls.append(("sfload", path))
        return 7

    def program_select(self, chan, sfid, bank, preset) -> None:
        self.calls.append(("program_select", chan, sfid, bank, preset))

    def noteon(self, chan, key, vel) -> None:
        self.calls.append(("noteon", chan, key, vel))

    def noteoff(self, chan, key) -> None:
        self.calls.append(("noteoff", chan, key))

    def get_samples(self, frames: int) -> np.ndarray:
        self.calls.append(("get_samples", frames))
        return np.full(frames * 2, 1000, dtype=np.int16)

    def delete(self) -> None:
        self.deleted = True


@pytest.fixture
def fake_fluidsynth(monkeypatch: pytest.MonkeyPatch) -> type[FakeSynth]:
    FakeSynth.instances.clear()
    monkeypatch.setitem(
        sys.modules, "fluidsynth", types.SimpleNamespace(Synth=FakeSynth)
    )

    return FakeSynth


@pytest.fixture
def soundfont_file(tmp_path: Path) -> Path:
    path = tmp_path / "dummy.sf2"
    path.write_bytes(b"placeholder")

    return path


def test_synthesize_returns_mono_float32(
    fake_fluidsynth: type[FakeSynth],
    soundfont_file: Path,
) -> None:
    synthesizer = Synthesizer(
        soundfont_file, sample_rate=48_000, release_tail_seconds=0.5
    )
    audio = synthesizer.synthesize(
        Melody((Note(60, 0.0, 1.0), Note(62, 2.0, 3.0)))
    )

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # 0.5s of first note + 0.5s rest + 0.5s second note + 0.5s tail at 48 kHz.
    assert audio.size == 96_000
    assert audio[0] == pytest.approx(1000 / 32768.0)
    assert fake_fluidsynth.instances[0].deleted


def test_synthesize_renders_between_events(
    fake_fluidsynth: type[FakeSynth],
    soundfont_file: Path,
) -> None:
    synthesizer = Synthesizer(
        soundfont_file,
        sample_rate=48_000,
        tempo=120.0,
        release_tail_seconds=0.5,
    )
    synthesizer.synthesize(Melody((Note(60, 0.0, 1.0), Note(62, 2.0, 3.0))))

    calls = fake_fluidsynth.instances[0].calls

    assert calls[0] == ("sfload", str(soundfont_file))
    assert calls[1] == ("program_select", 0, 7, 0, 0)
    assert calls[2:] == [
        ("noteon", 0, 60, 80),
        ("get_samples", 24_000),
        ("noteoff", 0, 60),
        ("get_samples", 24_000),
        ("noteon", 0, 62, 80),
        ("get_samples", 24_000),
        ("noteoff", 0, 62),
        ("get_samples", 24_000),
    ]


def test_adjacent_same_pitch_releases_before_rearticulating(
    fake_fluidsynth: type[FakeSynth],
    soundfont_file: Path,
) -> None:
    synthesizer = Synthesizer(soundfont_file, sample_rate=48_000)
    synthesizer.synthesize(Melody((Note(60, 0.0, 0.5), Note(60, 0.5, 1.0))))

    note_calls = [
        call[0]
        for call in fake_fluidsynth.instances[0].calls
        if call[0].startswith("note")
    ]

    assert note_calls == ["noteon", "noteoff", "noteon", "noteoff"]


def test_empty_melody_returns_empty_array_without_opening_synth(
    fake_fluidsynth: type[FakeSynth],
    soundfont_file: Path,
) -> None:
    synthesizer = Synthesizer(soundfont_file)
    audio = synthesizer.synthesize(Melody())

    assert audio.size == 0
    assert audio.dtype == np.float32
    assert fake_fluidsynth.instances == []


def test_missing_soundfont_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SoundFont not found"):
        Synthesizer(tmp_path / "missing.sf2")


def test_no_soundfont_configured_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOUNDFONT_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=SOUNDFONT_ENV_VAR):
        Synthesizer()


def test_soundfont_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    fake_fluidsynth: type[FakeSynth],
    soundfont_file: Path,
) -> None:
    monkeypatch.setenv(SOUNDFONT_ENV_VAR, str(soundfont_file))
    synthesizer = Synthesizer()

    assert synthesizer.soundfont_path == soundfont_file
    assert synthesizer.synthesize(Melody((Note(60, 0.0, 0.5),))).size > 0


def test_invalid_settings_rejected(soundfont_file: Path) -> None:
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, sample_rate=0)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, tempo=0)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, program=200)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, velocity=0)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, release_tail_seconds=-1)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, gain=0)
    with pytest.raises(ValueError):
        Synthesizer(soundfont_file, midi_channel=16)


@pytest.mark.integration
def test_real_fluidsynth_renders_dataset_melody(
    soundfont_path: Path,
    data_root: Path,
) -> None:
    example = next(iter(MelodyDataset(split="test", root=data_root)))
    synthesizer = Synthesizer(
        soundfont_path, sample_rate=22_050, release_tail_seconds=0.5
    )
    audio = synthesizer.synthesize(example.melody)

    expected_samples = round((example.melody.duration * 0.5 + 0.5) * 22_050)

    assert audio.dtype == np.float32
    assert abs(audio.size - expected_samples) <= 1
    assert np.max(np.abs(audio)) > 0.001
