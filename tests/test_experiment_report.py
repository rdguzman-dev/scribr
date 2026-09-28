"""Tests for JSON and Markdown experiment reports."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.common.evaluation import (
    ExampleEvaluation,
    ExperimentEvaluation,
)
from experiments.common.materialize import Manifest, ManifestExample
from experiments.common.predictions import TextPredictions
from experiments.common.report import write_report
from experiments.common.spec import (
    DatasetSpec,
    ExperimentSpec,
    SynthesisSpec,
)
from scribr.representation import Melody, Note
from scribr.synthesis import Instrument

METRICS = {
    "Precision": 1.0,
    "Recall": 1.0,
    "F-measure": 1.0,
    "Average_Overlap_Ratio": 1.0,
    "Precision_no_offset": 1.0,
    "Recall_no_offset": 1.0,
    "F-measure_no_offset": 1.0,
    "Average_Overlap_Ratio_no_offset": 1.0,
    "Onset_Precision": 1.0,
    "Onset_Recall": 1.0,
    "Onset_F-measure": 1.0,
    "Offset_Precision": 1.0,
    "Offset_Recall": 1.0,
    "Offset_F-measure": 1.0,
}


def example_instrument(index: int) -> Instrument:
    return Instrument.PIANO if index < 5 else Instrument.VIOLIN


def example_id(index: int) -> str:
    return f"{index:03d}-{example_instrument(index).value}"


def make_manifest() -> Manifest:
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
                id=example_id(index),
                index=index,
                candidate_index=index,
                instrument=example_instrument(index),
                program=example_instrument(index).program,
                wav=f"wav/{example_id(index)}.wav",
                reference=f"reference/{example_id(index)}.txt",
                wav_sha256="0" * 64,
                reference_sha256="0" * 64,
                num_notes=1,
            )
            for index in range(10)
        ),
    )


def make_evaluation() -> ExperimentEvaluation:
    per_example = []

    for index in range(10):
        instrument = example_instrument(index)
        metrics = dict(METRICS)

        if instrument is Instrument.VIOLIN:
            metrics["F-measure"] = 0.5
            metrics["F-measure_no_offset"] = 0.5

        per_example.append(
            ExampleEvaluation(
                example_id=example_id(index),
                group=instrument.value,
                metrics=metrics,
                off_grid_onsets=index % 2,
                off_grid_offsets=0,
            )
        )

    overall = dict(METRICS)
    overall["F-measure"] = 0.75
    overall["F-measure_no_offset"] = 0.75

    violin_metrics = dict(METRICS)
    violin_metrics["F-measure"] = 0.5
    violin_metrics["F-measure_no_offset"] = 0.5

    return ExperimentEvaluation(
        per_example=tuple(per_example),
        metrics=overall,
        grouped_metrics={
            "piano": dict(METRICS),
            "violin": violin_metrics,
        },
        tempo=120.0,
        metric_options={
            "onset_tolerance": 0.05,
            "pitch_tolerance": 50.0,
            "offset_ratio": 0.2,
            "offset_min_tolerance": 0.05,
        },
    )


def test_report_json_structure(tmp_path: Path) -> None:
    predictions = TextPredictions(
        melodies={"000-piano": Melody((Note(60, 0.0, 0.5),))},
        missing=("002-piano",),
        errors={"003-piano": "line 1: boom"},
    )

    json_path, markdown_path = write_report(
        make_manifest(),
        make_evaluation(),
        predictions,
        tmp_path,
        soundfont_path=tmp_path / "sound.sf2",
    )

    data = json.loads(json_path.read_text())

    assert data["experiment"] == "human_baseline"
    assert data["metadata"]["tempo"] == 120.0
    assert data["metadata"]["num_examples"] == 10
    assert data["metadata"]["soundfont_path"].endswith("sound.sf2")
    assert data["metadata"]["metric_options"]["onset_tolerance"] == 0.05
    assert set(data["metadata"]["package_versions"]) == {
        "scribr",
        "mir_eval",
    }
    assert data["overall"]["F-measure_no_offset"] == 0.75
    assert data["per_instrument"]["piano"]["num_examples"] == 5
    assert data["per_instrument"]["violin"]["metrics"]["F-measure"] == 0.5
    assert data["examples"][0] == {
        "id": "000-piano",
        "instrument": "piano",
        "off_grid_onsets": 0,
        "off_grid_offsets": 0,
        "metrics": METRICS,
    }
    assert data["missing"] == ["002-piano"]
    assert data["parse_errors"] == [
        {"id": "003-piano", "error": "line 1: boom"}
    ]
    assert markdown_path.is_file()


def test_markdown_tables_and_sections(tmp_path: Path) -> None:
    predictions = TextPredictions(
        melodies={},
        missing=("000-piano",),
        errors={"001-piano": "line 1: boom"},
    )

    _, markdown_path = write_report(
        make_manifest(),
        make_evaluation(),
        predictions,
        tmp_path,
        soundfont_path=tmp_path / "sound.sf2",
    )
    text = markdown_path.read_text()

    assert "## Overall metrics" in text
    assert "## Per-instrument metrics" in text
    assert "## Examples" in text
    assert "| F-measure_no_offset |" in text
    assert text.index("| F-measure_no_offset |") < text.index("| F-measure |")
    assert "release tails" in text
    assert "Per-instrument means are based on 5 examples each." in text
    assert "| Off-grid onsets | Off-grid offsets |" in text
    assert "## Missing transcriptions" in text
    assert "- `000-piano`" in text
    assert "## Parse errors" in text
    assert "- `001-piano`: line 1: boom" in text


def test_markdown_handles_empty_evaluation(tmp_path: Path) -> None:
    empty = ExperimentEvaluation(
        per_example=(),
        metrics={},
        grouped_metrics={},
        tempo=120.0,
        metric_options={
            "onset_tolerance": 0.05,
            "pitch_tolerance": 50.0,
            "offset_ratio": 0.2,
            "offset_min_tolerance": 0.05,
        },
    )

    _, markdown_path = write_report(
        make_manifest(),
        empty,
        TextPredictions(melodies={}, missing=(), errors={}),
        tmp_path,
    )
    text = markdown_path.read_text()

    assert "No examples were evaluated." in text
    assert "## Missing transcriptions" not in text
    assert "## Parse errors" not in text
