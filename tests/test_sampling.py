"""Tests for bounded-memory sampling helpers."""

from __future__ import annotations

import pytest

from scribr.data.sampling import reservoir_sample


def test_reservoir_sample_is_seeded_and_reads_the_whole_stream() -> None:
    items = range(100)

    first = reservoir_sample(items, 10, seed=42)
    second = reservoir_sample(items, 10, seed=42)
    other = reservoir_sample(items, 10, seed=0)

    assert len(first) == 10
    assert len(set(first)) == 10
    assert first == second
    assert first != other
    assert max(first) >= 10


def test_reservoir_sample_returns_everything_from_a_short_stream() -> None:
    assert sorted(reservoir_sample(range(3), 3, seed=42)) == [0, 1, 2]


@pytest.mark.parametrize("size", [0, 4])
def test_reservoir_sample_rejects_invalid_sizes(size: int) -> None:
    with pytest.raises(ValueError):
        reservoir_sample(range(3), size, seed=42)
