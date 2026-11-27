"""Evaluate Gemini on materialized WAVs, with a four-example default cap."""

import argparse
import importlib.metadata
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from experiments.common.materialize import Manifest, materialize
from experiments.common.predictions import TextPredictions
from experiments.common.report import write_report
from experiments.common.spec import ExperimentSpec
from experiments.common.transcription import load_recordings, score_recordings
from scribr.llm import LLMConfig, LLMTranscriber
from scribr.llm.table import PROMPT_VERSION
from scribr.representation import melody_to_text

_DIR = Path(__file__).resolve().parent


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--manifest", type=Path, help="reuse exact baseline WAVs"
    )
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=_DIR / "artifacts"
    )
    parser.add_argument("--max-examples", type=_positive_int, default=4)
    parser.add_argument("--model", help="override SCRIBR_LLM_MODEL")
    parser.add_argument("--unquantized", action="store_true")
    parser.add_argument("--max-attempts", type=_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = LLMConfig(
        model=args.model,
        quantize=not args.unquantized,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        min_interval_seconds=args.min_interval_seconds,
    )
    if not config.model:
        parser.error("set SCRIBR_LLM_MODEL or pass --model")

    if args.manifest:
        source_manifest = Manifest.load(args.manifest)
        source_dir = args.manifest.parent
    else:
        spec = ExperimentSpec.load(args.config or _DIR / "config.json")
        # Apply the spending cap before synthesis, even for full-split configs.
        size = spec.dataset.sample_size
        spec = replace(
            spec,
            dataset=replace(
                spec.dataset,
                sample_size=min(size, args.max_examples)
                if size
                else args.max_examples,
            ),
        )
        source_manifest = materialize(
            spec, args.artifacts_dir, soundfont_path=args.soundfont
        )
        source_dir = args.artifacts_dir

    manifest = replace(
        source_manifest,
        experiment="llm_gemini",
        examples=source_manifest.examples[: args.max_examples],
    )
    if not manifest.examples:
        parser.error("manifest must contain at least one example")
    cache_dir = (
        None
        if args.no_cache
        else (args.cache_dir or args.artifacts_dir / "responses")
    )
    transcriber = LLMTranscriber(config, cache_dir=cache_dir)
    try:
        evaluation, estimates = score_recordings(
            load_recordings(manifest, source_dir),
            transcriber,
            manifest.config.synthesis.tempo,
            verbose=True,
        )
    finally:
        transcriber.close()

    predictions_dir = args.artifacts_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for example_id, melody in estimates.items():
        (predictions_dir / f"{example_id}.txt").write_text(
            melody_to_text(melody), encoding="utf-8"
        )
    json_path, markdown_path = write_report(
        manifest,
        evaluation,
        TextPredictions(melodies=estimates, missing=(), errors={}),
        args.artifacts_dir,
        soundfont_path=args.soundfont,
        extra={
            "parameters": asdict(config),
            "source": {"manifest": source_manifest.to_dict()},
            "llm": {
                "provider": "gemini",
                "prompt_version": PROMPT_VERSION,
                "max_examples": args.max_examples,
                "selected_ids": [item.id for item in manifest.examples],
                "api_calls": transcriber.api_calls,
                "cache_hits": transcriber.cache_hits,
                "cache_enabled": cache_dir is not None,
            },
        },
        package_versions={
            "google-genai": importlib.metadata.version("google-genai"),
            "numpy": np.__version__,
        },
    )
    primary = evaluation.metrics["F-measure_no_offset"]
    print(f"Primary F-measure_no_offset: {primary:.4f}")
    print(f"Secondary F-measure: {evaluation.metrics['F-measure']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
