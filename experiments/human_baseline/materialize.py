"""Materialize the human baseline WAV and reference artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.materialize import materialize
from experiments.common.spec import ExperimentSpec

_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _DIR / "config.json"
_ARTIFACTS_DIR = _DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild artifacts even when a matching manifest exists",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="SoundFont path; defaults to $SCRIBR_SOUNDFONT",
    )
    args = parser.parse_args()

    spec = ExperimentSpec.load(_CONFIG_PATH)
    manifest = materialize(
        spec,
        _ARTIFACTS_DIR,
        soundfont_path=args.soundfont,
        force=args.force,
    )

    counts: dict[str, int] = {}

    for example in manifest.examples:
        instrument = example.instrument.value
        counts[instrument] = counts.get(instrument, 0) + 1

    print(
        f"Materialized {len(manifest.examples)} example(s) "
        f"in {_ARTIFACTS_DIR}"
    )

    for instrument, count in counts.items():
        print(f"  {instrument}: {count}")

    print(f"  wav: {_ARTIFACTS_DIR / 'wav'}")
    print(f"  reference: {_ARTIFACTS_DIR / 'reference'}")


if __name__ == "__main__":
    main()
