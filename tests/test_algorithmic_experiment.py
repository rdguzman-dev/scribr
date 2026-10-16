"""Verify baseline artifact loading and reproducible algorithmic reports."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.algorithmic import ablation
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
    manifest = Manifest.load(path)
    items = list(experiment.load_recordings(manifest, path.parent))
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
        list(experiment.load_recordings(manifest, path.parent))


def test_manifest_rejects_sample_rate_mismatch(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = Manifest.load(path)
    spec = replace(
        manifest.config,
        synthesis=replace(manifest.config.synthesis, sample_rate=44100),
    )
    mismatched = replace(manifest, config=spec)
    mismatched.save(path)
    with pytest.raises(ValueError, match="sample rate"):
        list(experiment.load_recordings(mismatched, path.parent))


def _stub_transcriber(reference, received=None):
    class Transcriber:
        def __init__(self, config):
            pass

        def transcribe(self, audio, sample_rate, tempo):
            if received is not None:
                received.append((audio.shape, sample_rate, tempo))

            return reference

    return Transcriber


def test_cli_writes_report_metrics_and_provenance(tmp_path, monkeypatch):
    path, reference = _manifest(tmp_path)
    received = []
    monkeypatch.setattr(
        experiment,
        "AlgorithmicTranscriber",
        _stub_transcriber(reference, received),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--manifest",
            str(path),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    experiment.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert received == [((22050,), 22050, 120)]
    assert report["metadata"]["num_examples"] == 1
    assert report["overall"]["F-measure_no_offset"] == 1
    assert report["per_instrument"]["piano"]["metrics"]["F-measure"] == 1
    assert report["source"]["manifest"] == Manifest.load(path).to_dict()
    assert report["parameters"]["min_pitch"] == 21
    assert "librosa" in report["metadata"]["package_versions"]
    assert (tmp_path / "report.md").is_file()


def test_cli_scores_recordings_lazily(tmp_path, monkeypatch):
    path, reference = _manifest(tmp_path)
    received = []
    real_score = experiment.score_recordings

    def spy_score(recordings, transcriber, tempo, *, verbose=False):
        received.append(recordings)
        return real_score(recordings, transcriber, tempo, verbose=verbose)

    monkeypatch.setattr(experiment, "score_recordings", spy_score)
    monkeypatch.setattr(
        experiment,
        "AlgorithmicTranscriber",
        _stub_transcriber(reference),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--manifest",
            str(path),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    experiment.main()

    # The full split must stream: do not decode every WAV up front.
    assert len(received) == 1
    assert not isinstance(received[0], list)
    assert iter(received[0]) is received[0]


def test_cli_materializes_config_and_scores(tmp_path, monkeypatch):
    path, reference = _manifest(tmp_path)
    manifest = Manifest.load(path)
    calls = []

    def fake_materialize(spec, artifacts_dir, *, soundfont_path=None):
        calls.append((spec, artifacts_dir, soundfont_path))
        return manifest

    config_path = tmp_path / "config.json"
    manifest.config.save(config_path)
    soundfont = tmp_path / "font.sf2"
    monkeypatch.setattr(
        experiment,
        "AlgorithmicTranscriber",
        _stub_transcriber(reference),
    )
    monkeypatch.setattr(experiment, "materialize", fake_materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            str(config_path),
            "--soundfont",
            str(soundfont),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    experiment.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert calls == [(manifest.config, tmp_path, soundfont)]
    assert report["source"]["manifest"] == manifest.to_dict()
    assert report["metadata"]["num_examples"] == 1


def test_baseline_tuning_and_validation_configs_use_separate_splits():
    directory = Path(experiment.__file__).parent
    baseline = ExperimentSpec.load(directory / "config.json")
    tuning = ExperimentSpec.load(directory / "tuning_config.json")
    validation = ExperimentSpec.load(directory / "validation_config.json")
    human = ExperimentSpec.load(
        directory.parent / "human_baseline" / "config.json"
    )

    assert baseline.dataset == DatasetSpec("test", None)
    assert human.dataset == DatasetSpec("test", 20)
    assert baseline.instruments == human.instruments
    assert baseline.synthesis == human.synthesis

    assert tuning.dataset == DatasetSpec("train", 2000)
    assert tuning.instruments == baseline.instruments
    assert tuning.synthesis == baseline.synthesis

    assert validation.dataset == DatasetSpec("validation", 2000)
    assert validation.instruments == tuning.instruments
    assert validation.synthesis == tuning.synthesis


class _QuantizeAwareTranscriber:
    """Return an off-grid estimate without quantization and an on-grid
    estimate with it, so the two ablation arms differ."""

    def __init__(self, config):
        self.config = config

    def transcribe(self, audio, sample_rate, tempo):
        if self.config.quantize:
            return Melody((Note(69, 0, 1),))

        return Melody((Note(69, 0.3, 1.3),))


def _ablation_result():
    recordings = [
        (
            "000-piano",
            Melody((Note(69, 0, 1),)),
            np.zeros(1, dtype=np.float32),
            22050,
            "piano",
        )
    ]

    return ablation.run_ablation(
        recordings,
        120,
        transcriber_factory=_QuantizeAwareTranscriber,
    )


def test_run_ablation_scores_both_arms():
    result = _ablation_result()
    assert [arm.arm.name for arm in result.arms] == [
        "unquantized",
        "quantized",
    ]
    assert result.winner == "quantized"
    assert result.scores == {"unquantized": 0.0, "quantized": 1.0}


def test_write_ablation_report_compares_arms(tmp_path):
    result = _ablation_result()
    json_path, markdown_path = ablation.write_ablation_report(
        result, artifacts_dir=tmp_path
    )
    data = json.loads(json_path.read_text())
    assert data["winner"] == "quantized"
    assert [arm["name"] for arm in data["arms"]] == [
        "unquantized",
        "quantized",
    ]
    assert data["arms"][0]["off_grid_onsets"] == 1
    assert data["arms"][1]["off_grid_onsets"] == 0
    assert data["difference"]["F-measure_no_offset"] == 1.0
    assert data["examples"][0]["primary_delta"] == 1.0
    markdown = markdown_path.read_text()
    assert "Higher mean `F-measure_no_offset`: quantized" in markdown


def test_ablation_cli_writes_comparison(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path)
    monkeypatch.setattr(
        ablation, "AlgorithmicTranscriber", _QuantizeAwareTranscriber
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ablation",
            "--manifest",
            str(path),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    ablation.main()
    data = json.loads((tmp_path / "ablation.json").read_text())
    assert data["winner"] == "quantized"
    assert (tmp_path / "ablation.md").is_file()
