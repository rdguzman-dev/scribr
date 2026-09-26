"""Tests for lazy split access and deterministic sampling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scribr.data import dataset as dataset_module
from scribr.data.dataset import DATA_ROOT_ENV_VAR, MelodyDataset


def test_iterates_all_shards_in_order(data_root: Path) -> None:
    dataset = MelodyDataset(split="train", root=data_root)

    assert [shard.name for shard in dataset.shards] == [
        "train_pitchseq-00000-of-00002.tfrecord",
        "train_pitchseq-00001-of-00002.tfrecord",
    ]

    examples = list(dataset)
    assert len(examples) == 12  # 8 + 4 records
    # First example comes from shard 0, ninth from shard 1.
    assert examples[0].pitch_sequence[:4] == (63, 128, 128, 64)
    assert examples[8].pitch_sequence[:4] == (129, 129, 55, 128)


def test_worker_shards_are_disjoint_and_complete(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordered = [e.pitch_sequence for e in MelodyDataset(split="train", root=data_root)]

    def collect(worker_id: int) -> list[tuple[int, ...]]:
        monkeypatch.setattr(
            dataset_module,
            "get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=2),
        )
        return [
            example.pitch_sequence
            for example in MelodyDataset(split="train", root=data_root)
        ]

    worker_0 = collect(0)
    worker_1 = collect(1)

    assert len(worker_0) == 8  # Shard 0
    assert len(worker_1) == 4  # Shard 1
    assert sorted(worker_0 + worker_1) == sorted(ordered)


def test_split_sizes(data_root: Path) -> None:
    assert len(list(MelodyDataset(split="validation", root=data_root))) == 8
    assert len(list(MelodyDataset(split="test", root=data_root))) == 4


def test_unknown_split_raises(data_root: Path) -> None:
    for split in ("dev", "val", "valid", "Train"):
        with pytest.raises(ValueError, match="unknown split"):
            MelodyDataset(split=split, root=data_root)


def test_missing_data_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        MelodyDataset(split="train", root=tmp_path)
    message = str(error.value)
    assert str(tmp_path / "train") in message
    assert "download script" in message
    assert "README.md" in message


def test_max_examples_limits_the_stream(data_root: Path) -> None:
    dataset = MelodyDataset(split="train", root=data_root, max_examples=5)
    examples = list(dataset)
    assert len(examples) == 5
    assert examples == list(MelodyDataset(split="train", root=data_root))[:5]


def test_max_examples_must_be_positive(data_root: Path) -> None:
    with pytest.raises(ValueError, match="max_examples"):
        MelodyDataset(split="train", root=data_root, max_examples=0)


def test_iteration_is_repeatable(data_root: Path) -> None:
    dataset = MelodyDataset(split="train", root=data_root, max_examples=8)
    first = [example.pitch_sequence for example in dataset]
    second = [example.pitch_sequence for example in dataset]
    assert first == second


def test_unshuffled_iteration_preserves_storage_order(data_root: Path) -> None:
    dataset = MelodyDataset(split="train", root=data_root, max_examples=6)
    expected = list(MelodyDataset(split="train", root=data_root))[:6]
    assert list(dataset) == expected


def test_seeded_shuffle_is_deterministic(data_root: Path) -> None:
    def collect(seed: int) -> list[tuple[int, ...]]:
        dataset = MelodyDataset(
            split="train",
            root=data_root,
            max_examples=12,
            shuffle=True,
            shuffle_buffer_size=12,
            seed=seed,
        )
        return [example.pitch_sequence for example in dataset]

    assert collect(42) == collect(42)
    assert collect(42) != collect(43)


def test_seeded_shuffle_is_a_permutation(data_root: Path) -> None:
    ordered = [e.pitch_sequence for e in MelodyDataset(split="train", root=data_root)]
    shuffled = [
        e.pitch_sequence
        for e in MelodyDataset(
            split="train", root=data_root, shuffle=True, shuffle_buffer_size=12, seed=1
        )
    ]
    assert sorted(shuffled) == sorted(ordered)


def test_shuffle_buffer_size_must_be_positive(data_root: Path) -> None:
    with pytest.raises(ValueError, match="shuffle_buffer_size"):
        MelodyDataset(split="train", root=data_root, shuffle_buffer_size=0)


def test_attributes_survive_iteration(data_root: Path) -> None:
    example = next(iter(MelodyDataset(split="validation", root=data_root)))
    assert example.attributes["note_density"] == pytest.approx(0.21875, abs=1e-6)
    assert len(example.attributes) == 13


def test_default_root_can_be_overridden_by_env(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(data_root))
    dataset = MelodyDataset(split="test")
    assert len(list(dataset)) == 4
