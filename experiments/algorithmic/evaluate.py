"""Evaluate the algorithmic transcriber on a split or a baseline manifest."""

import argparse
from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np

from experiments.common.json import read_json
from experiments.common.materialize import Manifest, materialize
from experiments.common.predictions import TextPredictions
from experiments.common.report import write_report
from experiments.common.spec import ExperimentSpec
from experiments.common.transcription import (
    Recording,
    load_recordings,
    score_recordings,
)
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber

_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--manifest",
        type=Path,
        help="score the exact materialized human-baseline audio",
    )
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument(
        "--parameters",
        type=Path,
        help="JSON mapping of AlgorithmicConfig parameters",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ARTIFACTS_DIR,
        help="directory for materialized artifacts and reports",
    )
    args = parser.parse_args()

    config = AlgorithmicConfig(
        **read_json(args.parameters) if args.parameters else {}
    )
    transcriber = AlgorithmicTranscriber(config)
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

    recordings = load_recordings(manifest, source_dir)
    evaluation, estimates = score_recordings(
        recordings, transcriber, spec.synthesis.tempo, verbose=True
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"manifest": manifest.to_dict()}

    json_path, markdown_path = write_report(
        manifest,
        evaluation,
        TextPredictions(melodies=estimates, missing=(), errors={}),
        artifacts_dir,
        soundfont_path=args.soundfont,
        extra={
            "parameters": asdict(config),
            "source": provenance,
        },
        package_versions={
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
    )

    print(
        f"F-measure_no_offset: {evaluation.metrics['F-measure_no_offset']:.4f}"
    )
    print(f"F-measure: {evaluation.metrics['F-measure']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
