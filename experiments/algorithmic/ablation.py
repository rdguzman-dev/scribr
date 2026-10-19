"""Compare grid-quantized and unquantized algorithmic transcriptions.

The same `AlgorithmicTranscriber` scores the same materialized audio
twice, once with `quantize=False` and once with the quantizing default.
Both arms share the audio, references, and every other parameter, so the
comparison is paired. The report lands in `ablation.json` and
`ablation.md`.

```bash
uv run python -m experiments.algorithmic.ablation \
  --manifest experiments/human_baseline/artifacts/manifest.json
```
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from experiments.common.markdown import display, format_score, markdown_table
from experiments.common.materialize import Manifest, materialize
from experiments.common.report import git_commit
from experiments.common.scoring import ExperimentEvaluation
from experiments.common.spec import ExperimentSpec
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber
from scribr.transcription import Transcriber

from .evaluate import Recording, load_recordings, score_recordings

_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _DIR / "artifacts"

PRIMARY_METRIC = "F-measure_no_offset"
SECONDARY_METRIC = "F-measure"


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One quantization setting in the comparison.

    Attributes:
        name: Short arm name used in run and report names.
        quantize: `AlgorithmicConfig.quantize` value the arm runs with.
    """

    name: str
    quantize: bool


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Scores for one arm.

    Attributes:
        arm: The quantization setting that produced the scores.
        evaluation: Batch evaluation for every example.
    """

    arm: AblationArm
    evaluation: ExperimentEvaluation


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Both arms and the comparison outcome.

    Attributes:
        parameters: Base config shared by both arms. The `quantize`
            field is overridden per arm.
        arms: One result per arm, unquantized first.
        primary_metric: Metric used to pick `winner`.
        secondary_metric: Offset-aware metric reported alongside.
        winner: Arm with the higher mean `primary_metric`, or `tie`.
    """

    parameters: AlgorithmicConfig
    arms: tuple[ArmResult, ...]
    primary_metric: str
    secondary_metric: str
    winner: str

    @property
    def scores(self) -> dict[str, float]:
        """Return mean primary scores keyed by arm name."""
        return {
            result.arm.name: result.evaluation.metrics[self.primary_metric]
            for result in self.arms
        }


_ARMS = (
    AblationArm(name="unquantized", quantize=False),
    AblationArm(name="quantized", quantize=True),
)


def run_ablation(
    recordings: Iterable[Recording],
    tempo: float,
    *,
    parameters: AlgorithmicConfig | None = None,
    transcriber_factory: (
        Callable[[AlgorithmicConfig], Transcriber] | None
    ) = None,
    verbose: bool = False,
) -> AblationResult:
    """Run both quantization arms on the same recordings.

    Args:
        recordings: Recordings to score. Both arms share them.
        tempo: Tempo in BPM shared by the recordings.
        parameters: Base config. Defaults to `AlgorithmicConfig()`. Only
            `quantize` differs between arms.
        transcriber_factory: Callable building a transcriber from a
            config, for tests. Defaults to `AlgorithmicTranscriber`.
        verbose: Print one line per transcribed example and arm.

    Returns:
        Per-arm evaluations and the primary-metric winner.
    """
    base = parameters if parameters is not None else AlgorithmicConfig()
    factory = (
        transcriber_factory
        if transcriber_factory is not None
        else AlgorithmicTranscriber
    )

    # Both arms iterate the recordings, so hold them once instead of
    # re-reading the artifact files for the second arm.
    recordings = list(recordings)
    results = []

    for arm in _ARMS:
        transcriber = factory(replace(base, quantize=arm.quantize))
        evaluation, _ = score_recordings(
            recordings, transcriber, tempo, verbose=verbose
        )
        results.append(ArmResult(arm=arm, evaluation=evaluation))

    return AblationResult(
        parameters=base,
        arms=tuple(results),
        primary_metric=PRIMARY_METRIC,
        secondary_metric=SECONDARY_METRIC,
        winner=_winner(results),
    )


def write_ablation_report(
    result: AblationResult,
    *,
    artifacts_dir: str | Path,
) -> tuple[Path, Path]:
    """Write `ablation.json` and `ablation.md` into `artifacts_dir`.

    Returns:
        The JSON and Markdown report paths.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    data = _report_dict(result)

    json_path = artifacts / "ablation.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    markdown_path = artifacts / "ablation.md"
    markdown_path.write_text(_markdown(data), encoding="utf-8")

    return json_path, markdown_path


def _winner(results: Sequence[ArmResult]) -> str:
    """Return the best arm name, or `tie` when the top scores match."""
    best = max(
        result.evaluation.metrics[PRIMARY_METRIC] for result in results
    )
    winners = [
        result.arm.name
        for result in results
        if result.evaluation.metrics[PRIMARY_METRIC] == best
    ]

    return winners[0] if len(winners) == 1 else "tie"


def _report_dict(result: AblationResult) -> dict[str, Any]:
    first = result.arms[0].evaluation
    arm_names = [arm.arm.name for arm in result.arms]

    return {
        "experiment": "algorithmic_pyin",
        "primary_metric": result.primary_metric,
        "secondary_metric": result.secondary_metric,
        "parameters": asdict(result.parameters),
        "metadata": {
            "git_commit": git_commit(),
            "python_version": platform.python_version(),
            "tempo": first.tempo,
            "num_examples": len(first.per_example),
            "metric_options": dict(first.metric_options),
        },
        "arms": [
            {
                "name": arm.arm.name,
                "quantize": arm.arm.quantize,
                "metrics": dict(arm.evaluation.metrics),
                "off_grid_onsets": _off_grid(arm, "off_grid_onsets"),
                "off_grid_offsets": _off_grid(arm, "off_grid_offsets"),
            }
            for arm in result.arms
        ],
        "per_instrument": {
            group: {
                arm.arm.name: dict(arm.evaluation.grouped_metrics[group])
                for arm in result.arms
            }
            for group in first.grouped_metrics
        },
        "examples": _example_records(result, arm_names),
        "difference": _difference(result),
        "winner": result.winner,
    }


def _off_grid(result: ArmResult, field: str) -> int:
    return sum(
        getattr(example, field) for example in result.evaluation.per_example
    )


def _example_records(
    result: AblationResult,
    arm_names: Sequence[str],
) -> list[dict[str, Any]]:
    by_arm = {
        arm.arm.name: {
            example.example_id: example
            for example in arm.evaluation.per_example
        }
        for arm in result.arms
    }
    records: list[dict[str, Any]] = []

    for example in result.arms[0].evaluation.per_example:
        scores = {}

        for name in arm_names:
            entry = by_arm[name][example.example_id]
            scores[name] = {
                result.primary_metric: entry.metrics[result.primary_metric],
                result.secondary_metric: entry.metrics[
                    result.secondary_metric
                ],
                "off_grid_onsets": entry.off_grid_onsets,
                "off_grid_offsets": entry.off_grid_offsets,
            }

        record: dict[str, Any] = {
            "id": example.example_id,
            "instrument": example.group,
            "scores": scores,
        }

        if len(arm_names) == 2:
            record["primary_delta"] = (
                scores[arm_names[1]][result.primary_metric]
                - scores[arm_names[0]][result.primary_metric]
            )

        records.append(record)

    return records


def _difference(result: AblationResult) -> dict[str, float]:
    """Return `quantized - unquantized` for the two reported metrics."""
    unquantized, quantized = result.arms

    return {
        result.primary_metric: (
            quantized.evaluation.metrics[result.primary_metric]
            - unquantized.evaluation.metrics[result.primary_metric]
        ),
        result.secondary_metric: (
            quantized.evaluation.metrics[result.secondary_metric]
            - unquantized.evaluation.metrics[result.secondary_metric]
        ),
    }


def _markdown(data: Mapping[str, Any]) -> str:
    metadata = data["metadata"]
    options = metadata["metric_options"]
    arms = data["arms"]
    arm_names = [arm["name"] for arm in arms]
    primary = data["primary_metric"]
    secondary = data["secondary_metric"]

    return "\n".join(
        [
            "# Algorithmic quantization ablation",
            "",
            "## Metadata",
            "",
            markdown_table(
                ("Field", "Value"),
                (
                    ("Git commit", display(metadata["git_commit"])),
                    ("Python", metadata["python_version"]),
                    ("Tempo (BPM)", str(metadata["tempo"])),
                    ("Examples", str(metadata["num_examples"])),
                    (
                        "Onset tolerance (s)",
                        str(options["onset_tolerance"]),
                    ),
                    (
                        "Pitch tolerance (cents)",
                        str(options["pitch_tolerance"]),
                    ),
                    ("Offset ratio", display(options["offset_ratio"])),
                    (
                        "Offset min tolerance (s)",
                        str(options["offset_min_tolerance"]),
                    ),
                ),
            ),
            "",
            "## Summary",
            "",
            markdown_table(
                (
                    "Arm",
                    "Quantize",
                    primary,
                    secondary,
                    "Off-grid onsets",
                    "Off-grid offsets",
                ),
                tuple(
                    (
                        arm["name"],
                        str(arm["quantize"]),
                        format_score(arm["metrics"][primary]),
                        format_score(arm["metrics"][secondary]),
                        str(arm["off_grid_onsets"]),
                        str(arm["off_grid_offsets"]),
                    )
                    for arm in arms
                ),
            ),
            "",
            (
                f"Quantized minus unquantized: "
                f"{format_score(data['difference'][primary])} `{primary}`, "
                f"{format_score(data['difference'][secondary])} `{secondary}`."
            ),
            "",
            f"Higher mean `{primary}`: {data['winner']}.",
            "",
            "## Per-instrument means",
            "",
            _per_instrument_table(data, arm_names),
            "",
            "## Per-example scores",
            "",
            _example_table(data, arm_names),
            "",
        ]
    )


def _per_instrument_table(
    data: Mapping[str, Any],
    arm_names: Sequence[str],
) -> str:
    primary = data["primary_metric"]

    return markdown_table(
        ("Instrument", *arm_names),
        tuple(
            (group,)
            + tuple(
                format_score(scores[name][primary]) for name in arm_names
            )
            for group, scores in data["per_instrument"].items()
        ),
    )


def _example_table(
    data: Mapping[str, Any],
    arm_names: Sequence[str],
) -> str:
    primary = data["primary_metric"]
    rows = []

    for example in data["examples"]:
        scores = example["scores"]
        rows.append(
            (example["id"], example["instrument"])
            + tuple(
                format_score(scores[name][primary]) for name in arm_names
            )
            + (_format_delta(example.get("primary_delta")),)
        )

    return markdown_table(("Example", "Instrument", *arm_names, "Delta"), rows)


def _format_delta(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--manifest",
        type=Path,
        help="compare on the exact materialized human-baseline audio",
    )
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument(
        "--parameters",
        type=Path,
        help=(
            "JSON mapping of shared AlgorithmicConfig parameters; "
            "`quantize` is set per arm"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ARTIFACTS_DIR,
        help="directory for materialized artifacts and reports",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print one line per transcribed example and arm",
    )
    args = parser.parse_args()
    parameters = AlgorithmicConfig(
        **json.loads(args.parameters.read_text()) if args.parameters else {}
    )
    artifacts_dir = args.artifacts_dir

    if args.manifest:
        manifest = Manifest.load(args.manifest)
        spec = manifest.config
        source_dir = args.manifest.parent

    else:
        spec = ExperimentSpec.load(args.config or _DIR / "config.json")
        manifest = materialize(
            spec, artifacts_dir, soundfont_path=args.soundfont
        )
        source_dir = artifacts_dir

    recordings = list(load_recordings(manifest, source_dir))
    result = run_ablation(
        recordings,
        spec.synthesis.tempo,
        parameters=parameters,
        verbose=args.verbose,
    )
    json_path, markdown_path = write_ablation_report(
        result, artifacts_dir=artifacts_dir
    )

    for arm_result in result.arms:
        primary = arm_result.evaluation.metrics[result.primary_metric]
        secondary = arm_result.evaluation.metrics[result.secondary_metric]
        print(
            f"{arm_result.arm.name}: {primary:.4f} {result.primary_metric}, "
            f"{secondary:.4f} {result.secondary_metric}"
        )

    print(f"Winner: {result.winner}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
