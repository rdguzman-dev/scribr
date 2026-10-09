"""Tests for deterministic artifact materialization."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.common.materialize import materialize
from experiments.common.spec import DatasetSpec, ExperimentSpec, SynthesisSpec
from scribr.data import MelodyDataset
from scribr.representation import (
    Melody,
    MelodyExample,
    Note,
    melody_from_text,
)
from scribr.synthesis import SOUNDFONT_ENV_VAR, Instrument

INSTRUMENTS = (
    Instrument.PIANO,
    Instrument.ACOUSTIC_GUITAR,
    Instrument.ELECTRIC_GUITAR,
    Instrument.VIOLIN,
)


class FakeSynthesizer:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.calls: list[Melody] = []

    def synthesize(self, melody: Melody) -> np.ndarray:
        self.calls.append(melody)
        frames = max(1, round(melody.duration * 100))

        return np.full(frames, self.value, dtype=np.float32)


class ListDataset:
    def __init__(self, examples: list[MelodyExample]) -> None:
        self.examples = examples

    def __iter__(self):
        return iter(self.examples)


def make_spec(
    *,
    prefix_size: int = 4,
    sample_size: int = 2,
    instruments: tuple[Instrument, ...] = INSTRUMENTS,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment="human_baseline",
        dataset=DatasetSpec(
            split="test",
            prefix_size=prefix_size,
            sample_size=sample_size,
        ),
        instruments=instruments,
        synthesis=SynthesisSpec(),
    )


def make_synthesizers(
    value: float = 0.25,
) -> dict[Instrument, FakeSynthesizer]:
    return {instrument: FakeSynthesizer(value) for instrument in INSTRUMENTS}


def make_examples(count: int) -> list[MelodyExample]:
    return [
        MelodyExample(Melody((Note(60 + index, 0.0, 1.0),)))
        for index in range(count)
    ]


def test_materialize_writes_manifest_wav_and_reference(
    tmp_path: Path,
    data_root: Path,
) -> None:
    spec = make_spec()
    dataset = MelodyDataset(split="test", root=data_root)
    candidates = list(dataset)

    manifest = materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(),
    )

    assert len(manifest.examples) == spec.dataset.sample_size
    assert manifest.soundfont_sha256 is None
    assert (tmp_path / "manifest.json").is_file()

    for example in manifest.examples:
        wav_path = tmp_path / example.wav
        reference_path = tmp_path / example.reference

        assert wav_path.is_file()
        assert reference_path.is_file()
        assert (
            example.wav_sha256
            == hashlib.sha256(wav_path.read_bytes()).hexdigest()
        )
        assert (
            example.reference_sha256
            == hashlib.sha256(reference_path.read_bytes()).hexdigest()
        )

        parsed = melody_from_text(reference_path.read_text())

        assert parsed == candidates[example.candidate_index].melody
        assert example.num_notes == len(parsed)
        assert example.program == example.instrument.program


def test_selection_and_round_robin_assignment(tmp_path: Path) -> None:
    dataset = ListDataset(make_examples(6))
    spec = make_spec(prefix_size=6, sample_size=5)

    manifest = materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(),
    )

    assert [example.candidate_index for example in manifest.examples] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [example.index for example in manifest.examples] == [0, 1, 2, 3, 4]
    assert [example.instrument for example in manifest.examples] == [
        Instrument.PIANO,
        Instrument.ACOUSTIC_GUITAR,
        Instrument.ELECTRIC_GUITAR,
        Instrument.VIOLIN,
        Instrument.PIANO,
    ]
    assert [example.id for example in manifest.examples] == [
        "000-piano",
        "001-acoustic_guitar",
        "002-electric_guitar",
        "003-violin",
        "004-piano",
    ]


def test_too_few_candidates_raises(tmp_path: Path) -> None:
    dataset = ListDataset(make_examples(2))
    spec = make_spec(prefix_size=4, sample_size=3)

    with pytest.raises(ValueError, match="only 2 candidate"):
        materialize(
            spec,
            tmp_path,
            dataset=dataset,
            synthesizers=make_synthesizers(),
        )


def test_two_runs_are_byte_identical(
    tmp_path: Path,
    data_root: Path,
) -> None:
    spec = make_spec()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = materialize(
        spec,
        first_dir,
        dataset=MelodyDataset(split="test", root=data_root),
        synthesizers=make_synthesizers(),
    )
    second = materialize(
        spec,
        second_dir,
        dataset=MelodyDataset(split="test", root=data_root),
        synthesizers=make_synthesizers(),
    )

    assert first == second
    assert (first_dir / "manifest.json").read_bytes() == (
        second_dir / "manifest.json"
    ).read_bytes()

    for example in first.examples:
        assert (first_dir / example.wav).read_bytes() == (
            second_dir / example.wav
        ).read_bytes()
        assert (first_dir / example.reference).read_bytes() == (
            second_dir / example.reference
        ).read_bytes()


def test_existing_manifest_is_reused(
    tmp_path: Path,
    data_root: Path,
) -> None:
    spec = make_spec()
    dataset = MelodyDataset(split="test", root=data_root)

    first = materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(0.25),
    )
    wav_before = (tmp_path / first.examples[0].wav).read_bytes()

    second_synthesizers = make_synthesizers(0.75)
    second = materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=second_synthesizers,
    )

    assert second == first
    assert (tmp_path / first.examples[0].wav).read_bytes() == wav_before
    assert all(not synth.calls for synth in second_synthesizers.values())


def test_force_rebuilds_artifacts(
    tmp_path: Path,
    data_root: Path,
) -> None:
    spec = make_spec()
    dataset = MelodyDataset(split="test", root=data_root)

    first = materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(0.25),
    )
    wav_before = (tmp_path / first.examples[0].wav).read_bytes()

    materialize(
        spec,
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(0.75),
        force=True,
    )

    assert (tmp_path / first.examples[0].wav).read_bytes() != wav_before


def test_different_config_without_force_raises(
    tmp_path: Path,
    data_root: Path,
) -> None:
    dataset = MelodyDataset(split="test", root=data_root)
    materialize(
        make_spec(sample_size=3),
        tmp_path,
        dataset=dataset,
        synthesizers=make_synthesizers(),
    )

    with pytest.raises(ValueError, match="different config"):
        materialize(
            make_spec(sample_size=2),
            tmp_path,
            dataset=dataset,
            synthesizers=make_synthesizers(),
        )


def test_missing_soundfont_raises_before_writing(
    tmp_path: Path,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOUNDFONT_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match=SOUNDFONT_ENV_VAR):
        materialize(
            make_spec(),
            tmp_path,
            dataset=MelodyDataset(split="test", root=data_root),
        )

    assert not (tmp_path / "manifest.json").exists()


def test_soundfont_path_builds_synthesizers_and_hashes_file(
    tmp_path: Path,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSynth:
        def __init__(self, *, gain: float, samplerate: float) -> None:
            self.gain = gain
            self.samplerate = samplerate

        def sfload(self, path: str) -> int:
            return 7

        def program_select(self, chan, sfid, bank, preset) -> None:
            pass

        def noteon(self, chan, key, vel) -> None:
            pass

        def noteoff(self, chan, key) -> None:
            pass

        def get_samples(self, frames: int) -> np.ndarray:
            return np.zeros(frames * 2, dtype=np.int16)

        def delete(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules, "fluidsynth", types.SimpleNamespace(Synth=FakeSynth)
    )

    soundfont = tmp_path / "test.sf2"
    soundfont.write_bytes(b"soundfont-bytes")

    manifest = materialize(
        make_spec(),
        tmp_path / "artifacts",
        dataset=MelodyDataset(split="test", root=data_root),
        soundfont_path=soundfont,
    )

    assert (
        manifest.soundfont_sha256
        == hashlib.sha256(b"soundfont-bytes").hexdigest()
    )
    assert [example.program for example in manifest.examples] == [0, 25]
