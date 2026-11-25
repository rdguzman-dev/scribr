"""Read and write JSON files in the project's on-disk format."""

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Load JSON from `path`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path) -> None:
    """Write `data` as two-space-indented JSON with a trailing newline."""
    text = json.dumps(data, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")
