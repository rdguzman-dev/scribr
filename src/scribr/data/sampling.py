"""Bounded-memory shuffling for streamed data.

`tfrecord.torch.TFRecordDataset` can shuffle with a bounded queue, but
its order depends on numpy's global RNG. `shuffle_buffer` provides a
reproducible shuffle from an explicit seed without materializing a
split.
"""

import random
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def shuffle_buffer(
    iterable: Iterable[T],
    *,
    buffer_size: int = 10_000,
    seed: int | None = None,
) -> Iterator[T]:
    """Shuffle a stream with a bounded memory buffer.

    Items are drawn uniformly from a buffer of `buffer_size` items.
    Memory is bounded by the buffer, and the same input order and seed
    always produce the same output order.
    """
    if buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")

    rng = random.Random(seed)
    buffer: list[T] = []

    for item in iterable:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue

        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = item

    rng.shuffle(buffer)

    yield from buffer
