# Scribr

Scribr explores algorithmic, supervised deep learning, and LLM-based
approaches to automated music transcription for quantized monophonic audio.

The repository currently contains the shared infrastructure for these
approaches, not the approaches themselves:

- `scribr.representation` defines the canonical `Note`, `Melody`, and
  `MelodyExample` types and the dataset pitch-sequence codec.
- `scribr.data` streams the TFRecord dataset on demand.
- `scribr.synthesis` renders a `Melody` to audio with FluidSynth.
- `scribr.midi` exports a `Melody` to a Standard MIDI File.

There is no transcription approach or evaluator yet.

## Setup

### Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- FluidSynth and a General MIDI SoundFont (only for audio synthesis)

`uv sync` installs PyTorch, numpy, mido, `tfrecord`, `pyfluidsynth`, and
pytest. TensorFlow is not required.

### 1. Install dependencies

```bash
git clone https://github.com/rdguzman-dev/scribr && cd scribr
uv sync
```

### 2. Download the dataset

The dataset is the 4 Bars Monophonic Melodies Dataset (PitchSequence), which
is not committed to this repository. The download script fetches the published
archives from [Zenodo](https://zenodo.org/records/13369389), verifies their
MD5 checksums, and extracts the shards into the layout `MelodyDataset`
expects:

```text
data/raw/4-bars-monophonic/
├── train/       # 8 shards, 10,126,676 melodies
├── validation/  # 8 shards, 70,908 melodies
└── test/        # 8 shards, 22,265 melodies
```

```bash
uv run python scripts/download_dataset.py                # all splits (~713 MB)
uv run python scripts/download_dataset.py --splits test  # test split (~1.7 MB)
```

Use `--dest` to extract somewhere else, then set `SCRIBR_DATA_ROOT` to that
directory. Shards already on disk are skipped, and `--force` re-downloads
them. The dataset is CC-BY-4.0.

### 3. Set up a SoundFont (synthesis only)

Synthesis needs the FluidSynth system library and a General MIDI SoundFont
(`.sf2` or `.sf3`). Install the library:

```bash
# macOS
brew install fluid-synth

# Debian/Ubuntu
sudo apt install libfluidsynth3
```

Then download a SoundFont such as
[GeneralUser GS](https://schristiancollins.com/generaluser.php) and point
Scribr at it:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUserGS.sf2
```

`Synthesizer(soundfont_path=...)` also accepts a path directly. On macOS,
Scribr sets `HOMEBREW_PREFIX` to `/opt/homebrew` when it is unset and the
Homebrew FluidSynth library is present there.

### 4. Verify the install

```bash
uv run pytest
uv run scribr  # prints a short banner if the package is importable
```

Unit tests use committed fixtures and a fake FluidSynth backend, so they pass
without the dataset or a SoundFont. The FluidSynth integration test is
skipped unless `SCRIBR_SOUNDFONT` is set:

```bash
uv run pytest -m integration
```

## Architecture

### Package layout

```text
src/scribr/
├── representation/  # Note, Melody, MelodyExample, pitch-sequence codec
├── data/            # lazy TFRecord access (MelodyDataset)
├── synthesis/       # Melody -> audio (Synthesizer)
└── midi/            # Melody -> .mid (write_midi)
scripts/download_dataset.py
tests/
```

### Data flow

```text
TFRecord shards
      │  scribr.data (lazy IterableDataset)
      ▼
MelodyExample ──► Melody  (canonical symbolic type)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   scribr.synthesis          scribr.midi
   Melody ──► audio          Melody ──► .mid
```

Future transcription approaches take audio and return a `Melody`, so
predictions and ground truth share one type.

The canonical in-memory transcription is a symbolic `Melody`, not a MIDI file
or a pitch-sequence token list. Time is measured in quarter-note beats from
the start of the melody, so `Note(pitch=60, onset=1.5, offset=1.75)` starts
1.5 beats in and lasts one 16th note. Ground truth and predictions use the
same type and the same units, so an evaluator can compare them directly.

### Dataset

`MelodyDataset` lazily streams one split through
`tfrecord.torch.TFRecordDataset` and returns one `MelodyExample` per record:

```python
from scribr.data import MelodyDataset

dataset = MelodyDataset(split="test")
example = next(iter(dataset))      # iterate; there is no indexing or len()
print(example.melody.notes[:3])    # (Note(pitch=63, onset=0.0, offset=0.75), ...)
print(example.attributes["note_density"])
print(example.pitch_sequence[:8])  # (63, 128, 128, 64, ...) pitch-sequence tokens
```

Each melody is 64 steps: 4 bars of 4/4, quantized to 4 steps per quarter note.
A step is a MIDI pitch (`21-108`), `128` (hold the current note), or `129` (no
note sounding). A new pitch token ends the current note, so consecutive
identical tokens decode to separate notes. The 13 attributes computed by the
dataset authors are preserved on `MelodyExample.attributes`.

`MelodyDataset` subclasses `torch.utils.data.IterableDataset`, so a
`DataLoader` can wrap it. Workers split the shard files round-robin and no
record is read twice. The split is never materialized: there is no index, no
manifest, and no `len()`, because counting records would require a full scan.

#### Reproducible subsets

The train split has 10 million melodies, so bound it with `max_examples` and
shuffle with a seeded bounded buffer:

```python
dataset = MelodyDataset(
    split="train",
    max_examples=100_000,  # per worker
    seed=42,               # deterministic shuffle order
    shuffle=True,
    shuffle_buffer_size=10_000,
)
```

`max_examples` limits the number of examples yielded per worker, so with
multiple `DataLoader` workers the total can be larger than `max_examples`. The
subset is drawn from the stored shard order, optionally shuffled. Both the
subset and the shuffle are deterministic for a given seed, memory stays bounded
by the buffer, and the official train/validation/test boundaries are always preserved.

### Synthesis

`Synthesizer` renders a `Melody` to a mono `float32` waveform in memory:

```python
from scribr.data import MelodyDataset
from scribr.synthesis import Synthesizer

example = next(iter(MelodyDataset(split="test")))
synthesizer = Synthesizer(sample_rate=22_050)  # or Synthesizer(soundfont_path="GeneralUserGS.sf2")
audio = synthesizer.synthesize(example.melody)
```

The waveform is a 1-D float32 array in [-1, 1]. Each call creates a fresh
FluidSynth instance, so release tails and controller state cannot leak from
one melody into the next, and nothing is written to disk. `soundfont_path`,
`sample_rate`, `tempo`, `program`, `velocity`, `gain`, `release_tail_seconds`,
and `midi_channel` are constructor arguments.

### MIDI export

MIDI is an export format only. Nothing in the dataset, synthesis, or future
evaluation path reads it. `write_midi` writes one monophonic track:

```python
from scribr.data import MelodyDataset
from scribr.midi import MidiExportConfig, write_midi

example = next(iter(MelodyDataset(split="test")))
write_midi(
    example.melody,
    "output.mid",
    MidiExportConfig(tempo=120.0, program=0),  # 0 = Acoustic Grand Piano
)
```

At the default 480 ticks per beat, one dataset step is 120 ticks. Adjacent
notes with the same pitch are released before being re-attacked, so repeated
pitches stay separate notes instead of merging.

### Future approaches

A transcription approach has the signature `audio -> Melody`. Evaluation can
compare its output to `MelodyExample.melody` directly, since both are the same
type in the same units:

```python
import numpy as np

from scribr.data import MelodyDataset
from scribr.representation import Melody
from scribr.synthesis import Synthesizer


def transcribe(audio: np.ndarray, sample_rate: int) -> Melody:
    raise NotImplementedError("human / algorithmic / DL / LLM approach")


dataset = MelodyDataset(split="validation", max_examples=100)
synthesizer = Synthesizer()

for example in dataset:
    audio = synthesizer.synthesize(example.melody)
    prediction = transcribe(audio, synthesizer.sample_rate)
    reference = example.melody
```

Each `Note` provides the `(pitch, onset, offset)` information needed by
note-level metrics such as those in `mir_eval`.
## Tests

```bash
uv run pytest                 # unit tests, no dataset or SoundFont needed
uv run pytest -m integration  # FluidSynth rendering (needs SCRIBR_SOUNDFONT)
```

Unit tests use byte-identical prefixes of real published shards
(`tests/fixtures/`) and a fake FluidSynth backend. The integration test is
skipped unless `SCRIBR_SOUNDFONT` points at an existing SoundFont.
