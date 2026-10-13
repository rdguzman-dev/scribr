"""Evaluate the algorithmic transcriber on a split or a baseline manifest."""

import argparse
import hashlib
import json
import wave
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scribr.data import MelodyDataset
from scribr.representation import Melody, melody_from_text

from experiments.common.materialize import Manifest
from experiments.common.scoring import evaluate_predictions
from experiments.common.spec import ExperimentSpec
from experiments.common.synthesis import (
    SynthesizerFactory,
    resolve_soundfont_path,
)
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber

_DIR = Path(__file__).resolve().parent


def _read_verified(path: Path, expected_hash: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"artifact hash mismatch: {path}")
    return data


def _manifest_audio(manifest_path: Path):
    manifest = Manifest.load(manifest_path)
    if not manifest.examples:
        raise ValueError("manifest must contain at least one example")
    for example in manifest.examples:
        wav_path = manifest_path.parent / example.wav
        _read_verified(wav_path, example.wav_sha256)
        with wave.open(str(wav_path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("manifest WAV files must be mono 16-bit PCM")
            sample_rate = source.getframerate()
            if sample_rate != manifest.config.synthesis.sample_rate:
                raise ValueError("WAV sample rate does not match manifest")
            audio = (
                np.frombuffer(
                    source.readframes(source.getnframes()), dtype="<i2"
                ).astype(np.float32)
                / 32768
            )
        reference = melody_from_text(
            _read_verified(
                manifest_path.parent / example.reference,
                example.reference_sha256,
            ).decode("utf-8")
        )
        yield (
            example.id,
            reference,
            audio,
            sample_rate,
            example.instrument.value,
        )


def _dataset_audio(spec: ExperimentSpec, soundfont: Path):
    factory = SynthesizerFactory(soundfont, spec.synthesis)
    synthesizers = {
        instrument: factory(instrument) for instrument in spec.instruments
    }
    dataset = MelodyDataset(
        split=spec.dataset.split, max_examples=spec.dataset.sample_size
    )
    count = 0
    for index, example in enumerate(dataset):
        instrument = spec.instruments[index % len(spec.instruments)]
        synthesizer = synthesizers[instrument]
        yield (
            f"{index:03d}-{instrument.value}",
            example.melody,
            synthesizer.synthesize(example.melody),
            synthesizer.sample_rate,
            instrument.value,
        )
        count += 1
    if count != spec.dataset.sample_size:
        raise ValueError(
            f"expected {spec.dataset.sample_size} examples, found {count}"
        )


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
        "--output", type=Path, default=_DIR / "artifacts" / "report.json"
    )
    args = parser.parse_args()
    config = AlgorithmicConfig(
        **json.loads(args.parameters.read_text()) if args.parameters else {}
    )
    transcriber = AlgorithmicTranscriber(config)
    if args.manifest:
        manifest = Manifest.load(args.manifest)
        spec = manifest.config
        recordings = _manifest_audio(args.manifest)
        provenance = {"manifest": manifest.to_dict()}
    else:
        spec = ExperimentSpec.load(args.config or _DIR / "config.json")
        soundfont = resolve_soundfont_path(args.soundfont)
        recordings = _dataset_audio(spec, soundfont)
        provenance = {
            "soundfont_sha256": hashlib.sha256(
                soundfont.read_bytes()
            ).hexdigest()
        }

    predictions: dict[str, list[dict]] = {}

    def items():
        for example_id, reference, audio, sample_rate, group in recordings:
            # References enter scoring only; inference receives audio and
            # the synthesis timing metadata, just like every other approach.
            estimate: Melody = transcriber.transcribe(
                audio, sample_rate, spec.synthesis.tempo
            )
            predictions[example_id] = [asdict(note) for note in estimate.notes]
            print(
                f"Transcribed {example_id}: {len(estimate)} note(s)",
                flush=True,
            )
            yield example_id, reference, estimate, group

    evaluation = evaluate_predictions(items(), tempo=spec.synthesis.tempo)
    report = {
        "approach": "algorithmic_pyin",
        "config": spec.to_dict(),
        "parameters": asdict(config),
        "versions": {
            name: version(name) for name in ("librosa", "numpy", "mir-eval")
        },
        "source": provenance,
        "num_examples": len(evaluation.per_example),
        "metrics": dict(evaluation.metrics),
        "grouped_metrics": dict(evaluation.grouped_metrics),
        "metric_options": dict(evaluation.metric_options),
        "per_example": [asdict(example) for example in evaluation.per_example],
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"F-measure_no_offset: {evaluation.metrics['F-measure_no_offset']:.4f}"
    )
    print(f"F-measure: {evaluation.metrics['F-measure']:.4f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
