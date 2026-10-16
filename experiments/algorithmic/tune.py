"""Automatic Optuna tuning for the algorithmic transcriber.

Each trial runs a complete `AlgorithmicTranscriber.transcribe` pass over a
screen subset of the tuning split. The best trials are re-scored on the
full tuning set, and the champion is scored on a held-out validation
split. Nothing is tuned or scored on the test split.

pYIN costs roughly 0.27 s per example, so scoring every trial on the full
2,000-example tuning set is expensive. Stage 1 screens every trial on a
500-example nested prefix; stage 2 re-ranks only the top finalists on the
full set. The winner is the stage-2 best, not `study.best_trial`.

```bash
uv run python -m experiments.algorithmic.tune \
  --config experiments/algorithmic/tuning_config.json \
  --validation-config experiments/algorithmic/validation_config.json \
  --trials 100 \
  --artifacts-dir experiments/algorithmic/artifacts/tuning
```
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import BaseSampler, TPESampler

from experiments.common.materialize import materialize
from experiments.common.spec import ExperimentSpec
from experiments.common.tracking import (
    DEFAULT_TRACKING_URI,
    git_metadata,
    log_artifact,
    log_metrics,
    log_params,
    start_run,
)
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber
from scribr.representation import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    Melody,
)

from .evaluate import load_recordings, score_recordings

_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _DIR / "artifacts"

EXPERIMENT = "algorithmic_tuning"
ANALYSIS_SAMPLE_RATE = 22_050
PRIMARY_METRIC = "F-measure_no_offset"
SECONDARY_METRIC = "F-measure"

FRAME_LENGTH_CHOICES = (1024, 2048, 4096)
HOP_LENGTH_CHOICES = (128, 256, 512)
RMS_WINDOW_HOP_CHOICES = (1, 2, 4, 8)
SILENCE_DB_RANGE = (-60.0, -20.0)
MIN_NOTE_SECONDS_RANGE = (0.02, 0.15)
MAX_GAP_SECONDS_RANGE = (0.0, 0.08)
ONSET_MERGE_SECONDS_RANGE = (0.0, 0.10)
ATTACK_REJECT_RATIO_RANGE = (0.0, 1.0)
ATTACK_BACKTRACK_SECONDS_RANGE = (0.0, 0.10)

Recording = tuple[str, Melody, np.ndarray, int, str]


@dataclass(frozen=True, slots=True)
class TunedTrial:
    """One completed stage-1 trial.

    Attributes:
        number: Optuna trial number.
        config: Validated config the trial suggested.
        score: Stage-1 mean `F-measure_no_offset`.
    """

    number: int
    config: AlgorithmicConfig
    score: float


@dataclass(frozen=True, slots=True)
class Finalist:
    """A promoted trial re-scored on the full tuning set.

    Attributes:
        trial: The stage-1 trial it came from.
        score_no_offset: Full-set mean `F-measure_no_offset`.
        score: Full-set mean `F-measure`.
    """

    trial: TunedTrial
    score_no_offset: float
    score: float


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Champion and every intermediate record from a tuning session.

    Attributes:
        champion: Winning config.
        champion_no_offset: Full-set `F-measure_no_offset` for `champion`.
        champion_score: Full-set `F-measure` for `champion`.
        finalists: Finalists ordered by stage-2 score.
        trials: Every completed stage-1 trial.
        validation_metrics: Metrics on the held-out validation split, or
            an empty mapping when no validation split was scored.
        best_parameters_path: JSON file with `champion`.
        trials_path: Stage-1 trials table.
        finalists_path: Stage-2 finalists table.
        validation_path: Validation metrics file, or `None`.
    """

    champion: AlgorithmicConfig
    champion_no_offset: float
    champion_score: float
    finalists: tuple[Finalist, ...]
    trials: tuple[TunedTrial, ...]
    validation_metrics: Mapping[str, float]
    best_parameters_path: Path
    trials_path: Path
    finalists_path: Path
    validation_path: Path | None


def _suggest_config(trial: optuna.Trial) -> AlgorithmicConfig:
    """Return a validated config built from suggested parameters.

    The dataset-specific fields are fixed: synthesis produced the audio
    at the analysis rate, the pitch bounds match the dataset range, and
    the evaluation only faces quantized audio.
    """
    return AlgorithmicConfig(
        sample_rate=ANALYSIS_SAMPLE_RATE,
        frame_length=trial.suggest_categorical(
            "frame_length", list(FRAME_LENGTH_CHOICES)
        ),
        hop_length=trial.suggest_categorical(
            "hop_length", list(HOP_LENGTH_CHOICES)
        ),
        min_pitch=DATASET_MIN_PITCH,
        max_pitch=DATASET_MAX_PITCH,
        silence_db=trial.suggest_float("silence_db", *SILENCE_DB_RANGE),
        min_note_seconds=trial.suggest_float(
            "min_note_seconds", *MIN_NOTE_SECONDS_RANGE, log=True
        ),
        max_gap_seconds=trial.suggest_float(
            "max_gap_seconds", *MAX_GAP_SECONDS_RANGE
        ),
        onset_merge_seconds=trial.suggest_float(
            "onset_merge_seconds", *ONSET_MERGE_SECONDS_RANGE
        ),
        quantize=True,
        attack_reject_ratio=trial.suggest_float(
            "attack_reject_ratio", *ATTACK_REJECT_RATIO_RANGE
        ),
        attack_backtrack_seconds=trial.suggest_float(
            "attack_backtrack_seconds", *ATTACK_BACKTRACK_SECONDS_RANGE
        ),
        rms_window_hops=trial.suggest_categorical(
            "rms_window_hops", list(RMS_WINDOW_HOP_CHOICES)
        ),
    )


def _default_trial_params() -> dict[str, Any]:
    """Return the suggested parameters for the default config."""
    config = AlgorithmicConfig()

    return {
        "frame_length": config.frame_length,
        "hop_length": config.hop_length,
        "silence_db": config.silence_db,
        "min_note_seconds": config.min_note_seconds,
        "max_gap_seconds": config.max_gap_seconds,
        "onset_merge_seconds": config.onset_merge_seconds,
        "attack_reject_ratio": config.attack_reject_ratio,
        "attack_backtrack_seconds": config.attack_backtrack_seconds,
        "rms_window_hops": config.rms_window_hops,
    }


def _search_space() -> dict[str, Any]:
    """Return the search space in JSON-compatible form, for run tags."""
    return {
        "sample_rate": ANALYSIS_SAMPLE_RATE,
        "min_pitch": DATASET_MIN_PITCH,
        "max_pitch": DATASET_MAX_PITCH,
        "quantize": True,
        "frame_length": list(FRAME_LENGTH_CHOICES),
        "hop_length": list(HOP_LENGTH_CHOICES),
        "silence_db": list(SILENCE_DB_RANGE),
        "min_note_seconds": {
            "low": MIN_NOTE_SECONDS_RANGE[0],
            "high": MIN_NOTE_SECONDS_RANGE[1],
            "log": True,
        },
        "max_gap_seconds": list(MAX_GAP_SECONDS_RANGE),
        "onset_merge_seconds": list(ONSET_MERGE_SECONDS_RANGE),
        "attack_reject_ratio": list(ATTACK_REJECT_RATIO_RANGE),
        "attack_backtrack_seconds": list(ATTACK_BACKTRACK_SECONDS_RANGE),
        "rms_window_hops": list(RMS_WINDOW_HOP_CHOICES),
    }


def _make_objective(
    recordings: Sequence[Recording],
    tempo: float,
):
    """Return the stage-1 Optuna objective over `recordings`."""

    def objective(trial: optuna.Trial) -> float:
        config = _suggest_config(trial)
        trial.set_user_attr("config", asdict(config))
        evaluation, _ = score_recordings(
            recordings, AlgorithmicTranscriber(config), tempo
        )
        trial.set_user_attr(
            PRIMARY_METRIC, evaluation.metrics[PRIMARY_METRIC]
        )
        trial.set_user_attr(
            SECONDARY_METRIC, evaluation.metrics[SECONDARY_METRIC]
        )

        return evaluation.metrics[PRIMARY_METRIC]

    return objective


def validate_limits(
    screen_examples: int,
    final_examples: int,
    top_k: int,
    total: int,
) -> None:
    """Raise `ValueError` when the two-stage sizes are inconsistent."""
    if screen_examples < 1:
        raise ValueError("screen_examples must be positive")

    if final_examples < screen_examples:
        raise ValueError("final_examples must be at least screen_examples")

    if final_examples > total:
        raise ValueError(
            f"final_examples must not exceed the tuning set size ({total})"
        )

    if top_k < 1:
        raise ValueError("top_k must be positive")


def promote_finalists(
    trials: Sequence[TunedTrial],
    *,
    top_k: int,
) -> tuple[TunedTrial, ...]:
    """Return the `top_k` stage-1 trials by score, highest first."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    return tuple(
        sorted(trials, key=lambda trial: trial.score, reverse=True)[:top_k]
    )


def select_champion(finalists: Sequence[Finalist]) -> Finalist:
    """Return the finalist with the best stage-2 `F-measure_no_offset`."""
    if not finalists:
        raise ValueError("at least one finalist is required")

    return max(finalists, key=lambda finalist: finalist.score_no_offset)


def _completed_trials(study: optuna.Study) -> tuple[TunedTrial, ...]:
    """Rebuild scored trials from the study's user attributes."""
    return tuple(
        TunedTrial(
            number=trial.number,
            config=AlgorithmicConfig(**trial.user_attrs["config"]),
            score=float(trial.value),
        )
        for trial in study.trials
        if trial.value is not None and "config" in trial.user_attrs
    )


def _log_screen_trial(
    trial: optuna.Trial,
    *,
    experiment: str,
    tracking_uri: str | Path,
    run_name: str,
) -> None:
    if trial.value is None or PRIMARY_METRIC not in trial.user_attrs:
        return

    with start_run(
        experiment,
        run_name=f"{run_name}-trial-{trial.number:04d}",
        tracking_uri=tracking_uri,
        tags={"stage": "screen", "trial": str(trial.number)},
        nested=True,
    ):
        log_params(trial.params)
        log_metrics(
            {
                PRIMARY_METRIC: float(trial.user_attrs[PRIMARY_METRIC]),
                SECONDARY_METRIC: float(trial.user_attrs[SECONDARY_METRIC]),
            }
        )


def _log_finalist(
    finalist: Finalist,
    *,
    experiment: str,
    tracking_uri: str | Path,
    run_name: str,
) -> None:
    with start_run(
        experiment,
        run_name=f"{run_name}-finalist-{finalist.trial.number:04d}",
        tracking_uri=tracking_uri,
        tags={"stage": "final", "trial": str(finalist.trial.number)},
        nested=True,
    ):
        log_params(asdict(finalist.trial.config))
        log_metrics(
            {
                PRIMARY_METRIC: finalist.score_no_offset,
                SECONDARY_METRIC: finalist.score,
            }
        )


def _write_finalists(path: Path, finalists: Sequence[Finalist]) -> None:
    if not finalists:
        raise ValueError("at least one finalist is required")

    config_fields = list(asdict(finalists[0].trial.config))
    fieldnames = [
        "trial",
        "screen_F-measure_no_offset",
        PRIMARY_METRIC,
        SECONDARY_METRIC,
        *config_fields,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for finalist in finalists:
            writer.writerow(
                {
                    "trial": finalist.trial.number,
                    "screen_F-measure_no_offset": finalist.trial.score,
                    PRIMARY_METRIC: finalist.score_no_offset,
                    SECONDARY_METRIC: finalist.score,
                    **asdict(finalist.trial.config),
                }
            )


def run_tuning(
    recordings: Sequence[Recording],
    tempo: float,
    *,
    trials: int | None,
    timeout: float | None,
    seed: int,
    n_jobs: int,
    screen_examples: int,
    final_examples: int | None,
    top_k: int,
    artifacts_dir: str | Path,
    tracking_uri: str | Path = DEFAULT_TRACKING_URI,
    run_name: str | None = None,
    dataset_config: Mapping[str, Any] | None = None,
    validation_recordings: Sequence[Recording] | None = None,
    sampler: BaseSampler | None = None,
) -> TuningResult:
    """Run one two-stage tuning session and log it to MLflow.

    Args:
        recordings: Tuning examples. The tuning spec selects a prefix, so
            the screen and final subsets are nested.
        tempo: Tempo in BPM shared by the recordings.
        trials: Number of stage-1 trials. `None` runs until `timeout`.
        timeout: Wall-clock limit in seconds for stage 1.
        seed: Seed for the default TPE sampler.
        n_jobs: Parallel stage-1 trials. Nested MLflow logging assumes
            serial execution.
        screen_examples: Stage-1 subset size.
        final_examples: Stage-2 subset size. Defaults to the tuning set.
        top_k: Number of stage-1 trials promoted to stage 2.
        artifacts_dir: Directory for the best parameters, trials table,
            finalists table, and validation metrics.
        tracking_uri: MLflow tracking store.
        run_name: Parent run name. Defaults to a fixed name.
        dataset_config: Dataset spec logged as a parent tag.
        validation_recordings: Held-out examples for the champion.
        sampler: Optuna sampler override, for tests.

    Returns:
        The champion and all recorded trial data.
    """
    recordings = list(recordings)

    if not recordings:
        raise ValueError("recordings must contain at least one example")

    total = len(recordings)

    if final_examples is None:
        final_examples = total

    validate_limits(screen_examples, final_examples, top_k, total)
    screen = recordings[:screen_examples]
    final = recordings[:final_examples]
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    name = run_name or "algorithmic-tuning"

    parent_tags = {
        **git_metadata(),
        "dataset": json.dumps(dataset_config or {}),
        "search_space": json.dumps(_search_space()),
        "screen_examples": str(screen_examples),
        "final_examples": str(final_examples),
        "top_k": str(top_k),
        "seed": str(seed),
    }

    with start_run(
        EXPERIMENT,
        run_name=name,
        tracking_uri=tracking_uri,
        tags=parent_tags,
    ):
        log_params(
            {
                "screen_examples": screen_examples,
                "final_examples": final_examples,
                "top_k": top_k,
                "seed": seed,
                "trials": trials,
                "timeout": timeout,
                "n_jobs": n_jobs,
            }
        )
        study = optuna.create_study(
            sampler=sampler if sampler is not None else TPESampler(seed=seed),
            direction="maximize",
        )
        study.enqueue_trial(_default_trial_params())

        def log_screen(study: optuna.Study, trial: optuna.Trial) -> None:
            _log_screen_trial(
                trial,
                experiment=EXPERIMENT,
                tracking_uri=tracking_uri,
                run_name=name,
            )

        study.optimize(
            _make_objective(screen, tempo),
            n_trials=trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=[log_screen],
        )

        completed = _completed_trials(study)
        finalist_trials = promote_finalists(completed, top_k=top_k)
        finalists: list[Finalist] = []

        for tuned in finalist_trials:
            evaluation, _ = score_recordings(
                final, AlgorithmicTranscriber(tuned.config), tempo
            )
            finalist = Finalist(
                trial=tuned,
                score_no_offset=evaluation.metrics[PRIMARY_METRIC],
                score=evaluation.metrics[SECONDARY_METRIC],
            )
            finalists.append(finalist)
            _log_finalist(
                finalist,
                experiment=EXPERIMENT,
                tracking_uri=tracking_uri,
                run_name=name,
            )

        ordered = tuple(
            sorted(
                finalists,
                key=lambda finalist: finalist.score_no_offset,
                reverse=True,
            )
        )
        champion = select_champion(ordered)

        validation_metrics: dict[str, float] = {}
        validation_path: Path | None = None

        if validation_recordings is not None:
            evaluation, _ = score_recordings(
                list(validation_recordings),
                AlgorithmicTranscriber(champion.trial.config),
                tempo,
            )
            validation_metrics = dict(evaluation.metrics)
            validation_path = artifacts / "validation_metrics.json"
            validation_path.write_text(
                json.dumps(validation_metrics, indent=2) + "\n",
                encoding="utf-8",
            )

        best_parameters_path = artifacts / "best_parameters.json"
        best_parameters_path.write_text(
            json.dumps(asdict(champion.trial.config), indent=2) + "\n",
            encoding="utf-8",
        )

        trials_path = artifacts / "trials_stage1.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

        finalists_path = artifacts / "finalists_stage2.csv"
        _write_finalists(finalists_path, ordered)

        with start_run(
            EXPERIMENT,
            run_name=f"{name}-summary",
            tracking_uri=tracking_uri,
            tags={"stage": "summary"},
            nested=True,
        ):
            log_params(asdict(champion.trial.config))
            metrics = {
                PRIMARY_METRIC: champion.score_no_offset,
                SECONDARY_METRIC: champion.score,
            }
            metrics.update(
                {
                    f"validation.{key}": value
                    for key, value in validation_metrics.items()
                }
            )
            log_metrics(metrics)
            log_artifact(best_parameters_path)
            log_artifact(trials_path)
            log_artifact(finalists_path)

            if validation_path is not None:
                log_artifact(validation_path)

    return TuningResult(
        champion=champion.trial.config,
        champion_no_offset=champion.score_no_offset,
        champion_score=champion.score,
        finalists=ordered,
        trials=completed,
        validation_metrics=validation_metrics,
        best_parameters_path=best_parameters_path,
        trials_path=trials_path,
        finalists_path=finalists_path,
        validation_path=validation_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DIR / "tuning_config.json",
        help="tuning split spec",
    )
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ARTIFACTS_DIR,
        help="directory for tuning artifacts and outputs",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel stage-1 trials; pYIN is CPU-bound",
    )
    parser.add_argument("--screen-examples", type=int, default=500)
    parser.add_argument(
        "--final-examples",
        type=int,
        help="stage-2 subset size; defaults to the whole tuning set",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING_URI))
    parser.add_argument("--run-name")
    parser.add_argument(
        "--validation-config",
        type=Path,
        help="held-out validation split spec",
    )
    args = parser.parse_args()

    spec = ExperimentSpec.load(args.config)
    artifacts_dir = args.artifacts_dir
    manifest = materialize(spec, artifacts_dir, soundfont_path=args.soundfont)
    recordings = list(load_recordings(manifest, artifacts_dir))

    validation_recordings = None

    if args.validation_config is not None:
        validation_spec = ExperimentSpec.load(args.validation_config)
        validation_dir = artifacts_dir / "validation"
        validation_manifest = materialize(
            validation_spec, validation_dir, soundfont_path=args.soundfont
        )
        validation_recordings = list(
            load_recordings(validation_manifest, validation_dir)
        )

    result = run_tuning(
        recordings,
        spec.synthesis.tempo,
        trials=args.trials,
        timeout=args.timeout,
        seed=args.seed,
        n_jobs=args.n_jobs,
        screen_examples=args.screen_examples,
        final_examples=args.final_examples,
        top_k=args.top_k,
        artifacts_dir=artifacts_dir,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
        dataset_config=spec.to_dict(),
        validation_recordings=validation_recordings,
    )

    print(f"Champion {PRIMARY_METRIC}: {result.champion_no_offset:.4f}")
    print(f"Champion {SECONDARY_METRIC}: {result.champion_score:.4f}")

    if result.validation_metrics:
        print(
            f"Validation {PRIMARY_METRIC}: "
            f"{result.validation_metrics[PRIMARY_METRIC]:.4f}"
        )
        print(
            f"Validation {SECONDARY_METRIC}: "
            f"{result.validation_metrics[SECONDARY_METRIC]:.4f}"
        )

    print("Champion parameters:")
    print(json.dumps(asdict(result.champion), indent=2))
    print(f"Wrote {result.best_parameters_path}")
    print(f"Wrote {result.trials_path}")
    print(f"Wrote {result.finalists_path}")

    if result.validation_path is not None:
        print(f"Wrote {result.validation_path}")


if __name__ == "__main__":
    main()
