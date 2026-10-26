"""Tests for experiment config parsing and validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.common.spec import (
    DatasetSpec,
    ExperimentSpec,
    SynthesisSpec,
)
from scribr.synthesis import Instrument

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "human_baseline"
    / "config.json"
)


def test_committed_config_loads() -> None:
    spec = ExperimentSpec.load(CONFIG_PATH)

    assert spec.experiment == "human_baseline"
    assert spec.dataset == DatasetSpec(split="test", sample_size=40, seed=42)
    assert spec.instruments == (
        Instrument.PIANO,
        Instrument.ACOUSTIC_GUITAR,
        Instrument.ELECTRIC_GUITAR,
        Instrument.VIOLIN,
    )
    assert spec.synthesis == SynthesisSpec()


def test_sample_size_none_selects_the_whole_split(tmp_path: Path) -> None:
    spec = ExperimentSpec.load(CONFIG_PATH)
    whole_split = replace(
        spec,
        dataset=DatasetSpec(split="test", sample_size=None),
    )
    path = tmp_path / "config.json"
    whole_split.save(path)

    assert ExperimentSpec.load(path).dataset.sample_size is None
    assert json.loads(path.read_text())["dataset"]["sample_size"] is None
    assert json.loads(path.read_text())["dataset"]["seed"] is None


def test_json_round_trip(tmp_path: Path) -> None:
    spec = ExperimentSpec.load(CONFIG_PATH)
    path = tmp_path / "config.json"

    spec.save(path)
    loaded = ExperimentSpec.load(path)

    assert loaded == spec
    assert json.loads(path.read_text()) == spec.to_dict()
    assert path.read_bytes() == CONFIG_PATH.read_bytes()
    assert spec.to_dict()["instruments"] == [
        "piano",
        "acoustic_guitar",
        "electric_guitar",
        "violin",
    ]
    assert spec.to_dict()["dataset"]["seed"] == 42


def test_dataset_validation_errors() -> None:
    with pytest.raises(ValueError, match="split"):
        DatasetSpec(split="dev")

    with pytest.raises(ValueError, match="sample_size"):
        DatasetSpec(sample_size=0)


def test_synthesis_validation_errors() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        SynthesisSpec(sample_rate=0)

    with pytest.raises(ValueError, match="tempo"):
        SynthesisSpec(tempo=0.0)

    with pytest.raises(ValueError, match="velocity"):
        SynthesisSpec(velocity=0)

    with pytest.raises(ValueError, match="velocity"):
        SynthesisSpec(velocity=128)

    with pytest.raises(ValueError, match="gain"):
        SynthesisSpec(gain=0.0)

    with pytest.raises(ValueError, match="release_tail_seconds"):
        SynthesisSpec(release_tail_seconds=-0.1)


def test_experiment_validation_errors() -> None:
    with pytest.raises(ValueError, match="instrument"):
        ExperimentSpec(
            experiment="x",
            dataset=DatasetSpec(),
            instruments=("cello",),
            synthesis=SynthesisSpec(),
        )

    with pytest.raises(ValueError, match="instrument"):
        ExperimentSpec(
            experiment="x",
            dataset=DatasetSpec(),
            instruments=(),
            synthesis=SynthesisSpec(),
        )

    with pytest.raises(ValueError, match="experiment"):
        ExperimentSpec(
            experiment="",
            dataset=DatasetSpec(),
            instruments=(Instrument.PIANO,),
            synthesis=SynthesisSpec(),
        )
