"""Tests for batch scoring, transcription loading, and the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.common.evaluation import evaluate_predictions
from experiments.common.materialize import Manifest, ManifestExample
from experiments.common.predictions import load_text_predictions
from experiments.common.spec import (
    DatasetSpec,
    ExperimentSpec,
    SynthesisSpec,
)
from experiments.human_baseline import evaluate as evaluate_cli
from scribr.representation import Melody, Note
from scribr.synthesis import Instrument


def make_manifest(ids: tuple[str, ...]) -> Manifest:
    return Manifest(
        experiment="human_baseline",
        soundfont_sha256=None,
        config=ExperimentSpec(
            experiment="human_baseline",
            dataset=DatasetSpec(),
            instruments=(Instrument.PIANO,),
            synthesis=SynthesisSpec(),
        ),
        examples=tuple(
            ManifestExample(
                id=example_id,
                index=index,
                candidate_index=index,
                instrument=Instrument.PIANO,
                program=0,
                wav=f"wav/{example_id}.wav",
                reference=f"reference/{example_id}.txt",
                wav_sha256="0" * 64,
                reference_sha256="0" * 64,
                num_notes=1,
            )
            for index, example_id in enumerate(ids)
        ),
    )


def test_perfect_and_imperfect_examples_average_overall() -> None:
    perfect = Melody((Note(60, 0.0, 0.5),))
    imperfect = Melody((Note(61, 0.0, 0.5),))

    result = evaluate_predictions(
        [
            ("a", perfect, perfect, "piano"),
            ("b", perfect, imperfect, "piano"),
        ],
        tempo=120.0,
    )

    assert result.metrics["F-measure"] == pytest.approx(0.5)
    assert result.metrics["F-measure_no_offset"] == pytest.approx(0.5)
    assert result.metrics["Onset_F-measure"] == 1.0
    assert [example.example_id for example in result.per_example] == ["a", "b"]


def test_grouped_means() -> None:
    perfect = Melody((Note(60, 0.0, 0.5),))
    imperfect = Melody((Note(61, 0.0, 0.5),))

    result = evaluate_predictions(
        [
            ("a", perfect, perfect, "piano"),
            ("b", perfect, imperfect, "violin"),
        ],
        tempo=120.0,
    )

    assert result.grouped_metrics["piano"]["F-measure_no_offset"] == 1.0
    assert result.grouped_metrics["violin"]["F-measure_no_offset"] == 0.0


def test_off_grid_counts_without_snapping() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))
    estimate = Melody((Note(60, 0.3, 0.75),))

    result = evaluate_predictions(
        [("a", reference, estimate, "piano")], tempo=120.0
    )

    example = result.per_example[0]

    assert example.off_grid_onsets == 1
    assert example.off_grid_offsets == 0
    assert estimate.notes[0].onset == 0.3


def test_empty_transcription_is_evaluated() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))

    with pytest.warns(UserWarning, match="Estimated notes are empty"):
        result = evaluate_predictions(
            [("a", reference, Melody(), "piano")], tempo=120.0
        )

    assert result.per_example[0].metrics["F-measure"] == 0.0
    assert result.per_example[0].metrics["F-measure_no_offset"] == 0.0


def test_empty_batch_returns_empty_aggregates() -> None:
    result = evaluate_predictions([], tempo=120.0)

    assert result.per_example == ()
    assert result.metrics == {}
    assert result.grouped_metrics == {}


def test_load_text_predictions(tmp_path: Path) -> None:
    manifest = make_manifest(("a", "b", "c"))
    (tmp_path / "a.txt").write_text("C4, 0.0, 0.5\n")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "c.txt").write_text("not a note table\n")

    predictions = load_text_predictions(manifest, tmp_path)

    assert predictions.melodies["a"] == Melody((Note(60, 0.0, 0.5),))
    assert predictions.melodies["b"] == Melody()
    assert predictions.missing == ()
    assert set(predictions.errors) == {"c"}
    assert "expected 3" in predictions.errors["c"]


def test_load_text_predictions_records_missing_files(tmp_path: Path) -> None:
    manifest = make_manifest(("a", "b"))

    predictions = load_text_predictions(manifest, tmp_path)

    assert predictions.missing == ("a", "b")
    assert predictions.melodies == {}
    assert predictions.errors == {}


def write_artifacts(tmp_path: Path) -> Manifest:
    manifest = make_manifest(("000-piano", "001-acoustic_guitar"))
    manifest.save(tmp_path / "manifest.json")
    (tmp_path / "reference").mkdir()

    for example in manifest.examples:
        (tmp_path / example.reference).write_text("C4, 0.0, 0.5\n")

    return manifest


def run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *arguments: str,
) -> None:
    transcriptions = tmp_path / "transcriptions"
    transcriptions.mkdir(exist_ok=True)
    monkeypatch.setattr(evaluate_cli, "_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate_cli, "_TRANSCRIPTIONS_DIR", transcriptions)
    monkeypatch.setattr(sys, "argv", ["evaluate", *arguments])

    evaluate_cli.main()


def test_cli_check_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "transcriptions").mkdir()
    (tmp_path / "transcriptions" / "000-piano.txt").write_text(
        "C4, 0.0, 0.5\n"
    )

    with pytest.raises(SystemExit) as exit_info:
        run_cli(tmp_path, monkeypatch, "--check")

    assert exit_info.value.code == 1

    output = capsys.readouterr().out

    assert "present: 1" in output
    assert "missing: 1" in output
    assert "parse errors: 0" in output


def test_cli_fails_on_incomplete_transcriptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "transcriptions").mkdir()
    (tmp_path / "transcriptions" / "000-piano.txt").write_text(
        "C4, 0.0, 0.5\n"
    )

    with pytest.raises(SystemExit) as exit_info:
        run_cli(tmp_path, monkeypatch)

    assert exit_info.value.code == 1
    assert "missing: 001-acoustic_guitar" in capsys.readouterr().out
    assert not (tmp_path / "report.json").exists()


def test_cli_partial_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "transcriptions").mkdir()
    (tmp_path / "transcriptions" / "000-piano.txt").write_text(
        "C4, 0.0, 0.5\n"
    )

    run_cli(tmp_path, monkeypatch, "--partial")

    report = json.loads((tmp_path / "report.json").read_text())

    assert report["metadata"]["num_examples"] == 1
    assert report["missing"] == ["001-acoustic_guitar"]
    assert report["overall"]["F-measure_no_offset"] == 1.0
    assert (tmp_path / "report.md").is_file()


def test_cli_complete_run_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    manifest = write_artifacts(tmp_path)
    (tmp_path / "transcriptions").mkdir()

    for example in manifest.examples:
        (tmp_path / "transcriptions" / f"{example.id}.txt").write_text(
            "C4, 0.0, 0.5\n"
        )

    run_cli(tmp_path, monkeypatch)

    output = capsys.readouterr().out

    assert "F-measure_no_offset: 1.0000" in output
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
