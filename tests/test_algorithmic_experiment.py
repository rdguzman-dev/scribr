"""Verify baseline artifact loading and reproducible algorithmic reports."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.algorithmic import evaluate as experiment
from experiments.common.materialize import Manifest, ManifestExample
from experiments.common.spec import DatasetSpec, ExperimentSpec, SynthesisSpec
from scribr.representation import Melody, Note, melody_to_text
from scribr.synthesis import Instrument
from scribr.wav import write_wav


def _manifest(tmp_path):
    reference = Melody((Note(69, 0, 1),))
    wav = tmp_path / "tone.wav"
    table = tmp_path / "reference.txt"
    write_wav(np.zeros(22050, dtype=np.float32), 22050, wav)
    table.write_text(melody_to_text(reference))
    spec = ExperimentSpec(
        "human_baseline",
        DatasetSpec("test", 1),
        (Instrument.PIANO,),
        SynthesisSpec(),
    )
    example = ManifestExample(
        "000-piano",
        0,
        Instrument.PIANO,
        0,
        wav.name,
        table.name,
        hashlib.sha256(wav.read_bytes()).hexdigest(),
        hashlib.sha256(table.read_bytes()).hexdigest(),
        1,
    )
    manifest = Manifest("human_baseline", None, spec, (example,))
    path = tmp_path / "manifest.json"
    manifest.save(path)
    return path, reference


def test_manifest_audio_is_verified_and_decoded(tmp_path):
    path, reference = _manifest(tmp_path)
    items = list(experiment._manifest_audio(path))
    assert len(items) == 1
    example_id, loaded, audio, sample_rate, instrument = items[0]
    assert (example_id, loaded, sample_rate, instrument) == (
        "000-piano",
        reference,
        22050,
        "piano",
    )
    assert audio.dtype == np.float32
    assert audio.shape == (22050,)
    (tmp_path / "reference.txt").write_text("C4, 0, 1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(experiment._manifest_audio(path))


def test_manifest_rejects_sample_rate_mismatch(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = Manifest.load(path)
    spec = replace(
        manifest.config,
        synthesis=replace(manifest.config.synthesis, sample_rate=44100),
    )
    replace(manifest, config=spec).save(path)
    with pytest.raises(ValueError, match="sample rate"):
        list(experiment._manifest_audio(path))


@pytest.mark.parametrize("unquantized", [False, True])
def test_cli_saves_predictions_metrics_and_provenance(
    tmp_path,
    monkeypatch,
    unquantized,
):
    path, reference = _manifest(tmp_path)
    received = []

    class Transcriber:
        def __init__(self, config):
            pass

        def transcribe(
            self,
            audio,
            sample_rate,
            tempo,
        ):
            received.append((audio.shape, sample_rate, tempo))
            return reference

    monkeypatch.setattr(experiment, "AlgorithmicTranscriber", Transcriber)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--manifest", str(path), "--output", str(output)]
        + (["--unquantized"] if unquantized else []),
    )
    experiment.main()
    report = json.loads(output.read_text())
    assert received == [((22050,), 22050, 120)]
    assert report["num_examples"] == 1
    assert report["metrics"]["F-measure_no_offset"] == 1
    assert report["grouped_metrics"]["piano"]["F-measure"] == 1
    assert report["predictions"]["000-piano"] == [
        {"pitch": 69, "onset": 0, "offset": 1}
    ]
    assert report["source"]["manifest"] == Manifest.load(path).to_dict()
    assert report["parameters"]["min_pitch"] == 21
    assert "librosa" in report["versions"]
    assert report["parameters"]["quantize"] is not unquantized


def test_test_baseline_and_training_tuning_configs_use_separate_splits():
    directory = Path(experiment.__file__).parent
    baseline = ExperimentSpec.load(directory / "config.json")
    tuning = ExperimentSpec.load(directory / "tuning_config.json")
    human = ExperimentSpec.load(
        directory.parent / "human_baseline" / "config.json"
    )

    assert baseline.dataset == human.dataset == DatasetSpec("test", 20)
    assert baseline.instruments == human.instruments
    assert baseline.synthesis == human.synthesis
    assert tuning.dataset.split == "train"
    assert tuning.dataset.sample_size == 1000
    assert tuning.instruments == baseline.instruments
    assert tuning.synthesis == baseline.synthesis
