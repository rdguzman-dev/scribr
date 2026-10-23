"""Score the hand-written human baseline transcriptions."""

from __future__ import annotations

import argparse
from pathlib import Path

from scribr.evaluation import (
    DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    DEFAULT_PITCH_TOLERANCE_CENTS,
)
from scribr.representation import melody_from_text

from experiments.common.materialize import Manifest
from experiments.common.predictions import (
    TextPredictions,
    load_text_predictions,
)
from experiments.common.report import write_report
from experiments.common.scoring import evaluate_predictions

_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _DIR / "artifacts"
_TRANSCRIPTIONS_DIR = _DIR / "transcriptions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report transcription progress without scoring",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="score the transcriptions that exist and list the rest",
    )
    args = parser.parse_args()

    manifest = Manifest.load(_ARTIFACTS_DIR / "manifest.json")
    predictions = load_text_predictions(manifest, _TRANSCRIPTIONS_DIR)

    if args.check:
        _print_check(predictions)
        raise SystemExit(
            1 if predictions.missing or predictions.errors else 0
        )

    if predictions.missing or predictions.errors:
        _print_problems(predictions)

        if not args.partial:
            print(
                "Transcriptions are incomplete; "
                "pass --partial to score now."
            )
            raise SystemExit(1)

    items = []

    for example in manifest.examples:
        estimate = predictions.melodies.get(example.id)

        if estimate is None:
            continue

        reference = melody_from_text(
            (_ARTIFACTS_DIR / example.reference).read_text(encoding="utf-8")
        )
        items.append(
            (example.id, reference, estimate, example.instrument.value)
        )

    evaluation = evaluate_predictions(
        items,
        tempo=manifest.config.synthesis.tempo,
        onset_tolerance=DEFAULT_ONSET_TOLERANCE_SECONDS,
        pitch_tolerance=DEFAULT_PITCH_TOLERANCE_CENTS,
        offset_ratio=DEFAULT_OFFSET_RATIO,
        offset_min_tolerance=DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
    )

    json_path, markdown_path = write_report(
        manifest, evaluation, predictions, _ARTIFACTS_DIR
    )

    print(
        f"Evaluated {len(evaluation.per_example)} of "
        f"{len(manifest.examples)} example(s)."
    )

    score = evaluation.metrics.get("F-measure_no_offset")

    if score is not None:
        print(f"F-measure_no_offset: {score:.4f}")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def _print_check(predictions: TextPredictions) -> None:
    print(f"present: {len(predictions.melodies)}")
    print(f"missing: {len(predictions.missing)}")
    print(f"parse errors: {len(predictions.errors)}")
    _print_problems(predictions)


def _print_problems(predictions: TextPredictions) -> None:
    for example_id in predictions.missing:
        print(f"missing: {example_id}")

    for example_id, message in predictions.errors.items():
        print(f"error: {example_id}: {message}")


if __name__ == "__main__":
    main()
