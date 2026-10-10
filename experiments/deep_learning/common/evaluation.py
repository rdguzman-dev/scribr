"""Score trained models with `mir_eval` and write run reports.

Scoring goes through `experiments.common.evaluation`, the same batch
scorer the human baseline uses, so numbers are comparable. Each
instrument renders the same references, and the report keeps overall
and per-instrument means plus the per-example rows.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.evaluation import (
    ExperimentEvaluation,
    evaluate_predictions,
    group_counts,
)
from experiments.common.markdown import display, format_score, markdown_table
from experiments.common.report import git_commit, metric_headers, metric_table
from scribr.deep_learning import (
    FeatureExtractor,
    SynthesizedMelodyDataset,
    melody_from_classes,
)
from scribr.representation import Melody, MelodyExample
from scribr.synthesis import Instrument, Synthesizer

from .spec import TrainingConfig


def run_evaluation(
    model: nn.Module,
    features: FeatureExtractor,
    device: torch.device,
    examples: Sequence[MelodyExample],
    config: TrainingConfig,
    synthesizer_factory: Callable[[Instrument], Synthesizer],
    *,
    batch_size: int,
    num_workers: int,
    decode: Callable[[Sequence[int]], Melody] = melody_from_classes,
) -> ExperimentEvaluation:
    """Transcribe `examples` with every evaluation instrument and score.

    The same reference melodies are rendered once per instrument, so the
    per-instrument split measures timbre generalization rather than a
    difference in the evaluation set.

    Args:
        model: Trained model.
        features: Feature extractor matching the model.
        device: Device to run inference on.
        examples: Reference melodies to transcribe.
        config: Training configuration, for the instrument list and
            synthesis tempo.
        synthesizer_factory: Callable returning a `Synthesizer` for one
            `Instrument`.
        batch_size: Inference batch size.
        num_workers: `DataLoader` workers for synthesis.
        decode: Maps per-step class indices to a `Melody`. Defaults to
            the dataset pitch-sequence codec.

    Returns:
        Per-example scores with overall and per-instrument means.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    items: list[tuple[str, Melody, Melody, str]] = []

    with torch.no_grad():
        for instrument in config.evaluation_instruments:
            dataset = SynthesizedMelodyDataset(
                examples,
                partial(synthesizer_factory, instrument),
                features,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            estimates = _estimates(loader, model, device, decode)

            for position, (example, estimate) in enumerate(
                zip(examples, estimates)
            ):
                items.append(
                    (
                        f"{position:03d}-{instrument.value}",
                        example.melody,
                        estimate,
                        instrument.value,
                    )
                )

    return evaluate_predictions(items, tempo=config.synthesis.tempo)


def write_report(
    evaluation: ExperimentEvaluation,
    config: TrainingConfig,
    *,
    split: str,
    checkpoint: str | Path,
    device: str,
    artifacts_dir: str | Path,
) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` into `artifacts_dir`.

    Returns:
        The JSON and Markdown report paths.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    counts = group_counts(evaluation)

    data = {
        "experiment": config.name,
        "split": split,
        "checkpoint": str(checkpoint),
        "metadata": {
            "git_commit": git_commit(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": device,
            "num_examples": len(evaluation.per_example),
            "tempo": evaluation.tempo,
            "metric_options": dict(evaluation.metric_options),
            "config": config.to_dict(),
        },
        "overall": dict(evaluation.metrics),
        "per_instrument": {
            instrument: {
                "num_examples": counts.get(instrument, 0),
                "metrics": dict(metrics),
            }
            for instrument, metrics in evaluation.grouped_metrics.items()
        },
        "examples": [
            {
                "id": example.example_id,
                "instrument": example.group,
                "off_grid_onsets": example.off_grid_onsets,
                "off_grid_offsets": example.off_grid_offsets,
                "metrics": dict(example.metrics),
            }
            for example in evaluation.per_example
        ],
    }

    json_path = artifacts / "report.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    markdown_path = artifacts / "report.md"
    markdown_path.write_text(_markdown(data), encoding="utf-8")

    return json_path, markdown_path


def _estimates(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    decode: Callable[[Sequence[int]], Melody],
) -> list[Melody]:
    melodies: list[Melody] = []

    for batch_features, _ in loader:
        logits = model(batch_features.to(device))

        for row in logits.argmax(dim=-1).cpu():
            melodies.append(decode(row.tolist()))

    return melodies


def _markdown(data: dict[str, Any]) -> str:
    metadata = data["metadata"]

    lines = [
        f"# {data['experiment']} report ({data['split']})",
        "",
        "## Metadata",
        "",
        markdown_table(
            ("Field", "Value"),
            (
                ("Git commit", display(metadata["git_commit"])),
                ("Python", metadata["python_version"]),
                ("Torch", metadata["torch_version"]),
                ("Device", metadata["device"]),
                ("Checkpoint", data["checkpoint"]),
                ("Examples evaluated", str(metadata["num_examples"])),
                ("Tempo (BPM)", str(metadata["tempo"])),
                (
                    "Onset tolerance (s)",
                    str(metadata["metric_options"]["onset_tolerance"]),
                ),
                (
                    "Pitch tolerance (cents)",
                    str(metadata["metric_options"]["pitch_tolerance"]),
                ),
            ),
        ),
        "",
        "## Overall metrics",
        "",
        metric_table(data["overall"]),
        "",
        "## Per-instrument metrics",
        "",
        _per_instrument_table(data["per_instrument"]),
        "",
    ]

    return "\n".join(lines)


def _per_instrument_table(per_instrument: dict[str, Any]) -> str:
    if not per_instrument:
        return "No examples were evaluated."

    headers = metric_headers(per_instrument.values())

    return markdown_table(
        ("Instrument", "N") + headers,
        tuple(
            (instrument, str(entry["num_examples"]))
            + tuple(
                format_score(entry["metrics"].get(name))
                for name in headers
            )
            for instrument, entry in per_instrument.items()
        ),
    )
