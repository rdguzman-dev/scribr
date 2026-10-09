"""Tests for shared SoundFont resolution and synthesizer construction."""

import pickle
from pathlib import Path

import pytest

from experiments.common.spec import SynthesisSpec
from experiments.common.synthesis import (
    SynthesizerFactory,
    resolve_soundfont_path,
)
from scribr.synthesis import SOUNDFONT_ENV_VAR, Instrument


def test_resolve_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SOUNDFONT_ENV_VAR, str(tmp_path / "environment.sf2"))
    explicit = tmp_path / "explicit.sf2"

    assert resolve_soundfont_path(explicit) == explicit


def test_resolve_falls_back_to_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soundfont = tmp_path / "environment.sf2"
    monkeypatch.setenv(SOUNDFONT_ENV_VAR, str(soundfont))

    assert resolve_soundfont_path(None) == soundfont


def test_resolve_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_soundfont_path("~/fonts.sf2") == tmp_path / "fonts.sf2"


def test_resolve_requires_a_soundfont(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOUNDFONT_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match=SOUNDFONT_ENV_VAR):
        resolve_soundfont_path(None)


def test_resolve_optional_reports_missing_soundfont(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOUNDFONT_ENV_VAR, raising=False)

    assert resolve_soundfont_path(None, required=False) is None


def test_factory_builds_each_instrument_with_synthesis_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    class Recorder:
        def __init__(self, soundfont_path: Path, **kwargs) -> None:
            calls.append((soundfont_path, kwargs))

    monkeypatch.setattr(
        "experiments.common.synthesis.Synthesizer", Recorder
    )
    synthesis = SynthesisSpec(
        sample_rate=16_000,
        tempo=90.0,
        velocity=64,
        gain=0.5,
        release_tail_seconds=0.25,
    )
    soundfont = Path("/soundfonts/test.sf2")
    factory = SynthesizerFactory(soundfont, synthesis)

    factory(Instrument.ACOUSTIC_GUITAR)

    assert calls == [
        (
            soundfont,
            {
                "sample_rate": 16_000,
                "tempo": 90.0,
                "program": Instrument.ACOUSTIC_GUITAR.program,
                "velocity": 64,
                "gain": 0.5,
                "release_tail_seconds": 0.25,
            },
        )
    ]


def test_factory_round_trips_through_pickle(tmp_path: Path) -> None:
    factory = SynthesizerFactory(
        soundfont_path=tmp_path / "test.sf2",
        synthesis=SynthesisSpec(sample_rate=16_000),
    )

    assert pickle.loads(pickle.dumps(factory)) == factory
