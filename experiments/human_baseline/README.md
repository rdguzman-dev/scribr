# Human baseline

This experiment measures how well a person can transcribe the 4 Bars
Monophonic Melodies dataset by ear. It has two phases. Phase 1 materializes
the first 20 test melodies as WAV files with reference note tables. Phase 2
parses the hand-written note tables and scores them against the references.

## Materialize

Materialization needs a General MIDI SoundFont. Point `SCRIBR_SOUNDFONT` at
one first:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.human_baseline.materialize
```

`--soundfont PATH` overrides the environment variable.

The command takes the first 20 examples from the test split and assigns the
four instruments round robin, 5 examples each. It writes:

```text
artifacts/
├── manifest.json
├── reference/<id>.txt
└── wav/<id>.wav
```

Running it again reuses the existing manifest; `--force` rebuilds
everything. The manifest records the config, the selected indices, and
file hashes. It holds no timestamps or filesystem paths, so the same
inputs always produce the same files. The WAV directory is gitignored; the
references, manifest, and transcriptions are committed.

## Transcribe

Each file in `artifacts/wav/` was transcribed manually by listening to the
audio and recording the notes heard in `transcriptions/<id>.txt`, using the ID
from the manifest (for example, `000-piano.txt`).

Each transcription uses one note per row in the following format:

```text
<pitch>, <onset>, <offset>
```

For example:

```text
C4, 0.0, 0.5
E4, 0.5, 1.0
G4, 1.0, 2.0
```

Each row represents one note, with its pitch, onset, and offset. Pitches use
scientific notation (`C4` is MIDI 60), and sharps and flats both parse. Onset
and offset are quarter-note beats from the start of the WAV. The header row is
optional. Times do not have to be aligned to the 0.25-beat grid. Scoring uses
the times exactly as written and reports how many rows fall off the grid.

### Transcription setup

The transcription was performed in GarageBand. A digital synthesizer capable of
emulating piano, acoustic guitar, electric guitar, and violin was available as a
reference instrument for identifying pitches by ear and using relative pitch.

Ordinary audio-inspection tools were permitted. The waveform could be
zoomed, navigated, looped, and segmented into smaller regions to make
difficult passages easier to hear. The timing ruler could also be used to
estimate note onsets and offsets. These tools were used to inspect and
navigate the audio, not to automatically extract musical information.

Automatic transcription or note-extraction tools were not used. In particular,
automatic pitch detection, onset or offset detection, MIDI transcription,
and other tools that directly infer note identities or timings from the audio
were prohibited.

The reference note tables in `artifacts/reference/` were not opened during
transcription. Those tables contain the ground truth, and reading them would
defeat the experiment.

The instrument release tail was ignored. A struck or bowed note may continue
sounding after its notated offset, but a fading tail was not transcribed as a
new note. The transcription represents the attacks that are heard, not the
decay.

### Timing log

The time spent on each transcription was recorded in `timing.csv`, one row
per example. The log is kept separate from the note tables in
`transcriptions/`, and the scoring pipeline does not read it.

Columns:

* `id`: example ID exactly as it appears in `transcriptions/` and the
  manifest (for example `000-piano`).
* `datetime`: date and time the transcription was started, in
  `YYYY-MM-DD HH:MM:SS` format.
* `duration_seconds`: how long the transcription took, recorded in seconds.
  Second-level precision is sufficient; small differences from the actual
  elapsed time are not meaningful.
* `notes`: optional free text for anything that explains outside influences
  or possible anomalies that affected the transcription time or accuracy
  (for example, being sick). Leave blank when nothing applies. Quote the
  field if it contains a comma.

An example row:

```text
001-acoustic_guitar,2026-09-28 18:00:00,2520,had a cold
```

## Score

Check progress at any time:

```bash
uv run python -m experiments.human_baseline.evaluate --check
```

`--check` prints the present, missing, and parse-error counts and exits 1 if
anything is missing or unparseable. It does not score.

Run the final evaluation when all 20 transcriptions exist:

```bash
uv run python -m experiments.human_baseline.evaluate
```

It writes `artifacts/report.json` and `artifacts/report.md` with overall and
per-instrument means, per-example metrics, off-grid counts, and environment
metadata. `F-measure_no_offset` is the headline metric because instrument
release tails make offset-aware scores noisy. To score an incomplete set
anyway, pass `--partial`; the report then lists the missing and unparseable
files instead of failing.
