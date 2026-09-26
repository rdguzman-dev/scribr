"""Download and extract the 4 Bars Monophonic Melodies dataset from
Zenodo.

The dataset is not committed to the repository. This script downloads the
published archives, verifies their MD5 checksums, and extracts the TFRecord
shards into the directory structure expected by `MelodyDataset`:

```text
data/raw/4-bars-monophonic/
├── train/
├── validation/
└── test/
```

Usage:

```text
uv run python scripts/download_dataset.py
uv run python scripts/download_dataset.py --splits test
```
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Archive:
    """Metadata for a dataset archive.

    Attributes:
        filename: Name of the archive file.
        md5: Expected MD5 checksum of the archive.
        size: Expected archive size in bytes.
    """

    filename: str
    md5: str
    size: int


# Zenodo record for the 4 Bars Monophonic Melodies dataset.
RECORD_ID = "13369389"
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"

# Official splits, keyed by the directory names under `data/raw`.
SPLITS: dict[str, Archive] = {
    "train": Archive(
        "4_bars_melodies_pitchseq_train.zip",
        "603ceb90d76c34b0b8ce439a61420677",
        705_917_056,
    ),
    "validation": Archive(
        "4_bars_melodies_pitchseq_validation.zip",
        "6a2c1abd19c4c606173e37a910aded86",
        5_325_881,
    ),
    "test": Archive(
        "4_bars_melodies_pitchseq_test.zip",
        "bdcf214aa1b3e2491f3e7ae98d8a0830",
        1_661_560,
    ),
}

# Default destination, relative to the current working directory.
DEFAULT_DEST = Path("data/raw/4-bars-monophonic")


def download_split(
    split: str,
    destination: Path,
    *,
    force: bool = False,
) -> Path:
    """Download and extract a dataset split.

    Args:
        destination: Dataset root; the split is extracted into
            `destination / split`.
        force: Re-extract the split, downloading the archive again if
            needed.

    Returns:
        The directory containing the extracted split.
    """
    archive = SPLITS[split]
    destination.mkdir(parents=True, exist_ok=True)
    split_dir = destination / split
    split_dir.mkdir(parents=True, exist_ok=True)

    if any(split_dir.glob("*.tfrecord")) and not force:
        print(f"[{split}] shards already present in {split_dir}, skipping")
        return split_dir

    downloads = destination / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / archive.filename
    url = f"{BASE_URL}/{archive.filename}/content"

    if archive_path.exists() and _md5(archive_path) == archive.md5:
        print(f"[{split}] using cached {archive_path}")

    else:
        print(f"[{split}] downloading {url}")
        _download(url, archive_path, archive.size)
        actual = _md5(archive_path)

        if actual != archive.md5:
            archive_path.unlink()
            raise SystemExit(
                f"[{split}] MD5 mismatch: expected {archive.md5}, got {actual}"
            )

    print(f"[{split}] extracting to {split_dir}")

    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(split_dir)

    return split_dir


def _download(url: str, destination: Path, expected_size: int) -> None:
    """Stream `url` to `destination`, printing progress as it goes."""
    with (
        urllib.request.urlopen(url) as response,
        destination.open("wb") as output,
    ):
        total = expected_size
        downloaded = 0

        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            percent = 100 * downloaded / total if total else 0
            print(
                f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB "
                f"({percent:5.1f}%)",
                end="",
                flush=True,
            )
    print()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and download the requested dataset splits."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLITS),
        default=sorted(SPLITS),
        help="splits to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"dataset root directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-extract even if shards already exist",
    )

    args = parser.parse_args(argv)

    for split in args.splits:
        download_split(split, args.dest, force=args.force)

    print(f"\nDone. Dataset root: {args.dest.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
