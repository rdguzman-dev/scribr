"""Load human note-table transcriptions for a materialized manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scribr.representation import Melody, melody_from_text

from .materialize import Manifest


@dataclass(frozen=True, slots=True)
class TextPredictions:
    """Transcriptions loaded from `<id>.txt` note tables.

    Attributes:
        melodies: Parsed transcription per example ID, in manifest
            order. An empty note table parses to an empty `Melody` and
            is a legitimate transcription.
        missing: Example IDs without a transcription file.
        errors: Parse error message per example ID.
    """

    melodies: Mapping[str, Melody]
    missing: tuple[str, ...]
    errors: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "melodies", dict(self.melodies))
        object.__setattr__(self, "errors", dict(self.errors))
        object.__setattr__(self, "missing", tuple(self.missing))


def load_text_predictions(
    manifest: Manifest,
    transcriptions_dir: str | Path,
) -> TextPredictions:
    """Parse one note-table file per manifest example.

    Missing files are recorded in `missing` rather than treated as an
    error. A file that exists but cannot be parsed is recorded in
    `errors` with its `ValueError` message.
    """
    directory = Path(transcriptions_dir)
    melodies: dict[str, Melody] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}

    for example in manifest.examples:
        path = directory / f"{example.id}.txt"

        if not path.is_file():
            missing.append(example.id)
            continue

        try:
            melodies[example.id] = melody_from_text(
                path.read_text(encoding="utf-8")
            )

        except ValueError as error:
            errors[example.id] = str(error)

    return TextPredictions(
        melodies=melodies,
        missing=tuple(missing),
        errors=errors,
    )
