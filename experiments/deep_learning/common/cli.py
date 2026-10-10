"""Command-line entry points shared by deep learning experiments.

Each experiment keeps a thin `train.py` and `evaluate.py` that pass its
feature and model builders into these functions, so the flags, config
overrides, and MLflow wiring stay in one place.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.common.tracking import (
    DEFAULT_TRACKING_URI,
    git_metadata,
    log_artifact,
    log_metrics,
    start_run,
)
from scribr.data import SPLITS
from scribr.deep_learning import FeatureExtractor

from .data import default_synthesizer_factory, load_examples
from .inference import run_evaluation, write_report
from .spec import DataSpec, TrainingConfig
from .training import EXPERIMENT, resolve_device, train

BuildFeatures = Callable[[Mapping[str, Any]], FeatureExtractor]
BuildModel = Callable[[Mapping[str, Any], Mapping[str, Any]], nn.Module]


def train_main(
    build_features: BuildFeatures,
    build_model: BuildModel,
    *,
    default_config: Path,
    default_artifacts_dir: Path,
    description: str | None = None,
) -> None:
    """Parse training arguments, train one model, and log the run."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="training configuration JSON",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name; defaults to the config name plus a timestamp",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="checkpoint directory; defaults to artifacts/<run name>",
    )
    _add_environment_arguments(parser)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch override, for smoke runs",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=None,
        help="training subset size override",
    )
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=None,
        help="validation subset size override",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="disable class weights",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=None,
        help=(
            "class weight exponent override; 1.0 is full inverse "
            "frequency, 0.5 is tempered"
        ),
    )
    args = parser.parse_args()

    config = TrainingConfig.load(args.config)

    if args.epochs is not None:
        config = replace(
            config,
            optimization=replace(config.optimization, epochs=args.epochs),
        )

    if args.train_examples is not None:
        config = replace(
            config,
            dataset=replace(config.dataset, max_examples=args.train_examples),
        )

    if args.validation_examples is not None:
        config = replace(
            config,
            validation=replace(
                config.validation,
                max_examples=args.validation_examples,
            ),
        )

    if args.no_class_weights:
        config = replace(
            config,
            optimization=replace(config.optimization, class_weights=False),
        )

    if args.class_weight_power is not None:
        config = replace(
            config,
            optimization=replace(
                config.optimization,
                class_weight_power=args.class_weight_power,
            ),
        )

    run_name = args.run_name or (
        f"{config.name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )
    artifacts_dir = args.artifacts_dir or default_artifacts_dir / run_name

    result = train(
        config,
        build_features=build_features,
        build_model=build_model,
        artifacts_dir=artifacts_dir,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        soundfont_path=args.soundfont,
        data_root=args.data_root,
        device=args.device,
    )

    print(f"Run: {result.run_name}")
    print(f"Artifacts: {result.artifacts_dir}")
    print(f"Best F-measure_no_offset: {result.best_metric:.4f}")


def evaluate_main(
    build_features: BuildFeatures,
    build_model: BuildModel,
    *,
    description: str | None = None,
) -> None:
    """Parse evaluation arguments, score one checkpoint, and log it."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint written by train.py",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="dataset split to score",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=512,
        help="number of leading records to score",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for synthesis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="inference batch size",
    )
    _add_environment_arguments(parser)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="report directory; defaults to the checkpoint directory",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name; defaults to the checkpoint and split",
    )
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = TrainingConfig.from_dict(payload["config"])
    resolved_device = resolve_device(args.device or config.optimization.device)

    features = build_features(config.features)
    model = build_model(config.model, config.features)
    model.load_state_dict(payload["model_state_dict"])

    examples = load_examples(
        DataSpec(
            split=args.split,
            max_examples=args.max_examples,
            num_workers=args.num_workers,
        ),
        root=args.data_root,
    )
    factory = default_synthesizer_factory(config, args.soundfont)

    evaluation = run_evaluation(
        model,
        features,
        resolved_device,
        examples,
        config,
        factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    run_name = args.run_name or (
        f"evaluate-{args.checkpoint.parent.name}-{args.split}"
    )
    artifacts_dir = args.artifacts_dir or args.checkpoint.parent
    json_path, markdown_path = write_report(
        evaluation,
        config,
        split=args.split,
        checkpoint=args.checkpoint,
        device=str(resolved_device),
        artifacts_dir=artifacts_dir,
    )

    tags = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "device": str(resolved_device),
        **git_metadata(),
    }

    with start_run(
        EXPERIMENT,
        run_name=run_name,
        tracking_uri=args.tracking_uri,
        tags=tags,
    ):
        log_metrics(
            {
                "eval/overall/F-measure_no_offset": evaluation.metrics.get(
                    "F-measure_no_offset", 0.0
                )
            }
        )

        for instrument, grouped in evaluation.grouped_metrics.items():
            log_metrics(
                {
                    f"eval/{instrument}/F-measure_no_offset": grouped.get(
                        "F-measure_no_offset", 0.0
                    )
                }
            )

        log_artifact(json_path)
        log_artifact(markdown_path)

    score = evaluation.metrics.get("F-measure_no_offset")

    print(
        f"Evaluated {len(evaluation.per_example)} example(s) on the "
        f"{args.split} split."
    )

    if score is not None:
        print(f"F-measure_no_offset: {score:.4f}")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that select the SoundFont, data, and tracking."""
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="SoundFont path; defaults to $SCRIBR_SOUNDFONT",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to the configured device",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dataset root; defaults to $SCRIBR_DATA_ROOT",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI or SQLite database path",
    )
