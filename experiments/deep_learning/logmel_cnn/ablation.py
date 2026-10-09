"""Compare class-weight schemes for the log-mel CNN.

Trains the same configuration twice, once with full inverse-frequency
weights and once with tempered weights, and compares the best
piano-only `F-measure_no_offset` on the validation subset. The tempered
arm is the new default; the inverse arm reproduces the original scheme
from before tempering. Both arms share the seed, data, and schedule, so
the comparison is paired.

```bash
uv run python -m experiments.deep_learning.logmel_cnn.ablation
```

The defaults are the 8-epoch diagnostic scale used for the earlier
weight comparison: 1,024 training melodies and 128 validation melodies.
Pass the full `config.json` sizes for a production comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from experiments.common.markdown import markdown_table
from experiments.common.tracking import (
    DEFAULT_TRACKING_URI,
    git_metadata,
    log_artifact,
    log_metrics,
    log_params,
    start_run,
)
from scribr.synthesis import Instrument, Synthesizer

from .spec import TrainingConfig
from .train import EXPERIMENT, TrainingResult, train

_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _DIR / "config.json"
_ARTIFACTS_DIR = _DIR / "artifacts"

# The ablation decides on piano-only scores so timbre transfer on the
# other instruments cannot sway the comparison.
PRIMARY_INSTRUMENT = Instrument.PIANO
PRIMARY_METRIC = "F-measure_no_offset"

_DIAGNOSTIC_EPOCHS = 8
_DIAGNOSTIC_TRAIN_EXAMPLES = 1024
_DIAGNOSTIC_VALIDATION_EXAMPLES = 128


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One class-weight scheme in the comparison.

    Attributes:
        name: Short arm name used in run and report names.
        power: `class_weight_power` value the arm trains with.
    """

    name: str
    power: float


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Best validation score for one arm.

    Attributes:
        arm: The weight scheme that was trained.
        run_name: MLflow run name.
        artifacts_dir: Directory holding the arm's checkpoints.
        best_epoch: Epoch with the best primary score.
        best_score: Best primary score.
        epoch_scores: `(epoch, score)` pairs for metric epochs.
    """

    arm: AblationArm
    run_name: str
    artifacts_dir: str
    best_epoch: int
    best_score: float
    epoch_scores: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Both arms and the comparison outcome.

    Attributes:
        run_name: Base name shared by the arm runs.
        metric: Primary metric label, `<instrument>/<metric>`.
        arms: One result per arm, in training order.
        winner: Name of the arm with the higher best score, or `tie`
            when the top scores are equal.
    """

    run_name: str
    metric: str
    arms: tuple[ArmResult, ...]
    winner: str

    @property
    def scores(self) -> dict[str, float]:
        """Return best primary scores keyed by arm name."""
        return {result.arm.name: result.best_score for result in self.arms}


def run_ablation(
    config: TrainingConfig,
    *,
    artifacts_dir: str | Path,
    run_name: str,
    tracking_uri: str | Path = DEFAULT_TRACKING_URI,
    soundfont_path: str | Path | None = None,
    data_root: str | Path | None = None,
    device: str | None = None,
    old_power: float = 1.0,
    new_power: float = 0.5,
    synthesizer_factory: Callable[[Instrument], Synthesizer] | None = None,
) -> AblationResult:
    """Train both weight schemes and pick the better primary score.

    Args:
        config: Base training configuration. Both arms share the
            dataset, model, and schedule; only
            `optimization.class_weight_power` changes. Each arm trains
            with `primary_instrument` set to piano.
        artifacts_dir: Root for the per-arm artifact directories.
        run_name: Base MLflow run name; each arm appends its name.
        tracking_uri: MLflow tracking URI, or a SQLite database path.
        soundfont_path: SoundFont override; defaults to
            `SCRIBR_SOUNDFONT`. Ignored when `synthesizer_factory` is
            given.
        data_root: Dataset root override, for tests and non-default
            checkouts.
        device: Device override; defaults to
            `config.optimization.device`.
        old_power: `class_weight_power` for the inverse-frequency arm.
        new_power: `class_weight_power` for the tempered arm.
        synthesizer_factory: Per-instrument synthesizer factory, for
            tests.

    Returns:
        Per-arm results and the primary-metric winner.
    """
    arms = (
        AblationArm(name="inverse", power=old_power),
        AblationArm(name="tempered", power=new_power),
    )
    artifacts = Path(artifacts_dir)
    results = tuple(
        _train_arm(
            arm,
            config,
            artifacts_dir=artifacts / arm.name,
            run_name=f"{run_name}-{arm.name}",
            tracking_uri=tracking_uri,
            soundfont_path=soundfont_path,
            data_root=data_root,
            device=device,
            synthesizer_factory=synthesizer_factory,
        )
        for arm in arms
    )
    winner = _winner(results)

    return AblationResult(
        run_name=run_name,
        metric=f"{PRIMARY_INSTRUMENT.value}/{PRIMARY_METRIC}",
        arms=results,
        winner=winner,
    )


def write_ablation_report(
    result: AblationResult,
    *,
    artifacts_dir: str | Path,
    config: TrainingConfig,
) -> tuple[Path, Path]:
    """Write `ablation.json` and `ablation.md` into `artifacts_dir`.

    Returns:
        The JSON and Markdown report paths.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    git_info = git_metadata()
    metadata = {
        "split": config.validation.split,
        "seed": config.optimization.seed,
        "epochs": config.optimization.epochs,
        "train_examples": config.dataset.max_examples,
        "validation_examples": config.validation.max_examples,
        "git_commit": git_info["git.commit"],
        "git_dirty": git_info["git.dirty"],
    }
    data = {
        "experiment": "logmel_cnn",
        "run_name": result.run_name,
        "primary_metric": result.metric,
        "metadata": metadata,
        "arms": [
            {
                "name": arm_result.arm.name,
                "power": arm_result.arm.power,
                "run_name": arm_result.run_name,
                "artifacts_dir": arm_result.artifacts_dir,
                "best_epoch": arm_result.best_epoch,
                "best_score": arm_result.best_score,
                "epoch_scores": [
                    [epoch, score] for epoch, score in arm_result.epoch_scores
                ],
            }
            for arm_result in result.arms
        ],
        "winner": result.winner,
    }

    json_path = artifacts / "ablation.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    markdown_path = artifacts / "ablation.md"
    markdown_path.write_text(_markdown(result, metadata), encoding="utf-8")

    return json_path, markdown_path


def _winner(results: Sequence[ArmResult]) -> str:
    """Return the best arm name, or `tie` when the top scores match."""
    best = max(result.best_score for result in results)
    winners = [
        result.arm.name for result in results if result.best_score == best
    ]

    return winners[0] if len(winners) == 1 else "tie"


def _train_arm(
    arm: AblationArm,
    config: TrainingConfig,
    *,
    artifacts_dir: Path,
    run_name: str,
    tracking_uri: str | Path,
    soundfont_path: str | Path | None,
    data_root: str | Path | None,
    device: str | None,
    synthesizer_factory: Callable[[Instrument], Synthesizer] | None,
) -> ArmResult:
    arm_config = replace(
        config,
        optimization=replace(
            config.optimization,
            class_weights=True,
            class_weight_power=arm.power,
            primary_instrument=PRIMARY_INSTRUMENT.value,
        ),
    )
    result = train(
        arm_config,
        artifacts_dir=artifacts_dir,
        run_name=run_name,
        tracking_uri=tracking_uri,
        soundfont_path=soundfont_path,
        data_root=data_root,
        device=device,
        synthesizer_factory=synthesizer_factory,
    )
    epoch_scores = _epoch_scores(result)
    best_epoch, best_score = max(epoch_scores, key=lambda item: item[1])

    return ArmResult(
        arm=arm,
        run_name=result.run_name,
        artifacts_dir=str(artifacts_dir),
        best_epoch=best_epoch,
        best_score=best_score,
        epoch_scores=epoch_scores,
    )


def _epoch_scores(
    result: TrainingResult,
) -> tuple[tuple[int, float], ...]:
    """Return `(epoch, primary score)` pairs from the training history."""
    scores: list[tuple[int, float]] = []

    for record in result.history:
        metrics = record.get("per_instrument", {}).get(
            PRIMARY_INSTRUMENT.value, {}
        )
        score = metrics.get(PRIMARY_METRIC)

        if score is None or not math.isfinite(score):
            continue

        scores.append((int(record["epoch"]), float(score)))

    if not scores:
        raise ValueError("training history has no primary metric scores")

    return tuple(scores)


def _markdown(
    result: AblationResult,
    metadata: Mapping[str, object],
) -> str:
    lines = [
        "# logmel_cnn class-weight ablation",
        "",
        "## Metadata",
        "",
        markdown_table(
            ("Field", "Value"),
            (
                ("Run", result.run_name),
                ("Primary metric", result.metric),
                ("Split", str(metadata["split"])),
                ("Seed", str(metadata["seed"])),
                ("Epochs", str(metadata["epochs"])),
                ("Train examples", str(metadata["train_examples"])),
                ("Validation examples", str(metadata["validation_examples"])),
                ("Git commit", str(metadata["git_commit"])),
                ("Git dirty", str(metadata["git_dirty"])),
            ),
        ),
        "",
        "## Summary",
        "",
        markdown_table(
            ("Arm", "Power", "Best epoch", "Best score"),
            tuple(
                (
                    arm_result.arm.name,
                    f"{arm_result.arm.power:.2f}",
                    str(arm_result.best_epoch),
                    f"{arm_result.best_score:.4f}",
                )
                for arm_result in result.arms
            ),
        ),
        "",
        f"Winner: {result.winner}.",
        "",
        "## Per-epoch scores",
        "",
        _score_table(result),
        "",
    ]

    return "\n".join(lines)


def _score_table(result: AblationResult) -> str:
    """Render one row per epoch with a column per arm."""
    names = [arm_result.arm.name for arm_result in result.arms]
    scores = {
        arm_result.arm.name: dict(arm_result.epoch_scores)
        for arm_result in result.arms
    }
    epochs = sorted(
        {epoch for by_epoch in scores.values() for epoch in by_epoch}
    )
    rows: list[tuple[str, ...]] = []

    for epoch in epochs:
        row = [str(epoch)]

        for name in names:
            score = scores[name].get(epoch)
            row.append(f"{score:.4f}" if score is not None else "")

        rows.append(tuple(row))

    return markdown_table(("Epoch", *names), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="base training configuration JSON",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="base MLflow run name; defaults to a timestamped name",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="output root; defaults to artifacts/<run name>",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="SoundFont path; defaults to $SCRIBR_SOUNDFONT",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dataset root; defaults to $SCRIBR_DATA_ROOT",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to the configured device",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_DIAGNOSTIC_EPOCHS,
        help="epoch override; defaults to the diagnostic 8",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=_DIAGNOSTIC_TRAIN_EXAMPLES,
        help="training subset size override; defaults to the diagnostic 1024",
    )
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=_DIAGNOSTIC_VALIDATION_EXAMPLES,
        help="validation subset size override; defaults to the diagnostic 128",
    )
    parser.add_argument(
        "--old-power",
        type=float,
        default=1.0,
        help="class weight exponent for the inverse-frequency arm",
    )
    parser.add_argument(
        "--new-power",
        type=float,
        default=0.5,
        help="class weight exponent for the tempered arm",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI or SQLite database path",
    )
    args = parser.parse_args()

    config = TrainingConfig.load(args.config)
    config = replace(
        config,
        dataset=replace(config.dataset, max_examples=args.train_examples),
        validation=replace(
            config.validation, max_examples=args.validation_examples
        ),
        optimization=replace(config.optimization, epochs=args.epochs),
    )
    run_name = args.run_name or (
        f"class-weight-ablation-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )
    artifacts_dir = args.artifacts_dir or _ARTIFACTS_DIR / run_name

    result = run_ablation(
        config,
        artifacts_dir=artifacts_dir,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        soundfont_path=args.soundfont,
        data_root=args.data_root,
        device=args.device,
        old_power=args.old_power,
        new_power=args.new_power,
    )
    json_path, markdown_path = write_ablation_report(
        result,
        artifacts_dir=artifacts_dir,
        config=config,
    )

    with start_run(
        EXPERIMENT,
        run_name=f"{run_name}-summary",
        tracking_uri=args.tracking_uri,
        tags={
            "ablation": "class-weights",
            "metric": result.metric,
            **git_metadata(),
        },
    ):
        log_params(
            {
                "old_power": args.old_power,
                "new_power": args.new_power,
                "epochs": config.optimization.epochs,
                "dataset.max_examples": config.dataset.max_examples,
                "validation.max_examples": config.validation.max_examples,
            }
        )

        for arm_result in result.arms:
            log_metrics(
                {
                    f"{arm_result.arm.name}/{PRIMARY_METRIC}": (
                        arm_result.best_score
                    )
                }
            )

        log_artifact(json_path)
        log_artifact(markdown_path)

    for arm_result in result.arms:
        print(
            f"{arm_result.arm.name}: {arm_result.best_score:.4f} "
            f"({result.metric}, epoch {arm_result.best_epoch})"
        )

    print(f"Winner: {result.winner}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
