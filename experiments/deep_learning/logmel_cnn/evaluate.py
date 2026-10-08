"""Evaluate a trained checkpoint on a dataset split.

Loads the config from the checkpoint, so the model and features are
rebuilt exactly as they were trained. Scoring covers every evaluation
instrument and writes `report.json` and `report.md` next to the
checkpoint, then records the summary in a separate MLflow run tagged
with the split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.common.tracking import (
    DEFAULT_TRACKING_URI,
    git_metadata,
    log_artifact,
    log_metrics,
    start_run,
)
from scribr.data import SPLITS

from .data import (
    build_features,
    build_model,
    default_synthesizer_factory,
    load_examples,
)
from .evaluation import run_evaluation, write_report
from .spec import DataSpec, TrainingConfig
from .train import EXPERIMENT, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--tracking-uri",
        type=Path,
        default=DEFAULT_TRACKING_URI,
        help="local MLflow store",
    )
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = TrainingConfig.from_dict(payload["config"])
    resolved_device = resolve_device(args.device or config.optimization.device)

    features = build_features(config.features)
    model = build_model(config.model, n_mels=features.n_mels)
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


if __name__ == "__main__":
    main()
