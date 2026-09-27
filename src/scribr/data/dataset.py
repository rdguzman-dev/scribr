"""Lazy access to the 4 Bars Monophonic Melodies TFRecord dataset.

The dataset is distributed as sharded TFRecord files of TensorFlow
`SequenceExample` records. `MelodyDataset` streams the shards of one
split through `tfrecord.torch.TFRecordDataset` and yields one
`MelodyExample` per record. Nothing is materialized at construction time
and no index or manifest is built; iteration reads the shards in sorted
filename order and decodes each record on demand.

Because `MelodyDataset` subclasses `torch.utils.data.IterableDataset`, a
`torch.utils.data.DataLoader` can wrap it directly. Workers split the
shard files round-robin, so no record is yielded twice; workers beyond
the shard count yield nothing. `max_examples` is applied per worker: a
`DataLoader` with `num_workers=N` streams up to `N * max_examples`
examples.
"""

import os
from collections.abc import Iterator
from pathlib import Path

from tfrecord.torch import TFRecordDataset
from torch.utils.data import IterableDataset, get_worker_info

from ..representation.melody import MelodyExample
from .example import (
    CONTEXT_DESCRIPTION,
    SEQUENCE_DESCRIPTION,
    melody_example_from_features,
)
from .sampling import shuffle_buffer

# Official splits, using the directory names under `data/raw`.
SPLITS: tuple[str, ...] = ("train", "validation", "test")

# Environment variable that overrides the dataset location.
DATA_ROOT_ENV_VAR = "SCRIBR_DATA_ROOT"

# `<repo>/data/raw/4-bars-monophonic` when running from a source checkout.
_DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "4-bars-monophonic"
)


class MelodyDataset(IterableDataset):
    """A lazy `torch.utils.data.IterableDataset` over one dataset split.

    Example:
        >>> dataset = MelodyDataset(
        ...     split="train",
        ...     max_examples=100_000,
        ...     seed=42,
        ... )
        >>> for example in dataset:  # doctest: +SKIP
        ...     audio = synthesizer.synthesize(example.melody)

    Args:
        split: One of `"train"`, `"validation"` or `"test"`.
        root: Dataset root containing the split directories. Defaults to
            `$SCRIBR_DATA_ROOT` or `data/raw/4-bars-monophonic`.
        max_examples: If given, limit each worker's stream to this many
            examples before optional shuffling. The cap is per worker,
            not global: a `DataLoader` with `num_workers=N` streams up
            to `N * max_examples` examples, fewer when a worker runs out
            of records. The subset each worker yields is a prefix of the
            record in its assigned shards; official split boundaries are
            always preserved.
        seed: Seed for the optional shuffle. Subsetting itself is
            deterministic; the seed only controls shuffle order.
        shuffle: Shuffle the selected subset with `shuffle_buffer`
            before yielding.
        shuffle_buffer_size: Size of the bounded shuffle window,
            independent of the subset size.

    Note:
        There is no `len()`: the number of records in a split is not
        known without a full scan, and scanning millions of records just
        to answer `len()` would defeat lazy iteration. Use
        `max_examples` to bound experiments, or
        `itertools.islice` for ad-hoc prefixes.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        root: str | Path | None = None,
        max_examples: int | None = None,
        seed: int | None = None,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10_000,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(
                f"unknown split {split!r}; expected one of {', '.join(SPLITS)}"
            )

        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be at least 1")

        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")

        self.split = split
        self.max_examples = max_examples
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        self.root = (
            Path(root).expanduser()
            if root is not None
            else default_data_root()
        )

        split_dir = self.root / self.split
        self._shards: tuple[Path, ...] = tuple(
            sorted(split_dir.glob("*.tfrecord"))
        )

        if not self._shards:
            raise FileNotFoundError(
                _missing_split_message(split_dir, self.split)
            )

    @property
    def shards(self) -> tuple[Path, ...]:
        """The shard files backing this split, in iteration order."""
        return self._shards

    def __repr__(self) -> str:
        return (
            f"MelodyDataset(split={self.split!r}, root={str(self.root)!r}, "
            f"shards={len(self._shards)}, max_examples={self.max_examples!r}, "
            f"seed={self.seed!r}, shuffle={self.shuffle!r})"
        )

    def _shards_in_this_process(self) -> tuple[Path, ...]:
        """Split the shards round-robin over `DataLoader` workers."""
        worker = get_worker_info()

        if worker is None:
            return self._shards

        return tuple(
            shard
            for index, shard in enumerate(self._shards)
            if index % worker.num_workers == worker.id
        )

    def _stream(self) -> Iterator[MelodyExample]:
        emitted = 0

        for shard in self._shards_in_this_process():
            reader = TFRecordDataset(
                str(shard),
                None,
                description=CONTEXT_DESCRIPTION,
                sequence_description=SEQUENCE_DESCRIPTION,
            )

            for context, features in reader:
                if (
                    self.max_examples is not None
                    and emitted >= self.max_examples
                ):
                    return

                yield melody_example_from_features(context, features)

                emitted += 1

    def __iter__(self) -> Iterator[MelodyExample]:
        """Iterate over the split lazily and sequentially."""
        stream = self._stream()

        if self.shuffle:
            return shuffle_buffer(
                stream,
                buffer_size=self.shuffle_buffer_size,
                seed=self.seed,
            )

        return stream


def default_data_root() -> Path:
    """Return the configured dataset root.

    `SCRIBR_DATA_ROOT` takes precedence over the repository-local
    `data/raw/4-bars-monophonic` directory.
    """
    configured = os.environ.get(DATA_ROOT_ENV_VAR)

    if configured:
        return Path(configured).expanduser()

    return _DEFAULT_DATA_ROOT


def _missing_split_message(split_dir: Path, split: str) -> str:
    return (
        f"No .tfrecord shards found in {split_dir} for the {split!r} split.\n"
        "Download the dataset and place the shards in the expected directory, "
        "or run the dataset download script. See README.md for details."
    )
