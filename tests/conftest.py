"""Shared test fixtures for Scribr's infrastructure tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_head() -> Path:
    """Byte-identical prefix (8 records) of a real published test shard."""
    return FIXTURES / "melodies_head.tfrecord"


@pytest.fixture
def data_root() -> Path:
    """Miniature dataset root with train/validation/test split directories."""
    return FIXTURES / "data"


@pytest.fixture
def soundfont_path() -> Path:
    """Path to a SoundFont for FluidSynth integration tests.

    Tests using this fixture are skipped unless ``SCRIBR_SOUNDFONT`` points
    at an existing ``.sf2``/``.sf3`` file.
    """
    configured = os.environ.get("SCRIBR_SOUNDFONT")
    if not configured:
        pytest.skip("set SCRIBR_SOUNDFONT to run FluidSynth integration tests")
    path = Path(configured).expanduser()
    if not path.is_file():
        pytest.skip(f"SCRIBR_SOUNDFONT does not exist: {path}")
    return path
