# Human baseline

This experiment measures how well a person can transcribe the 4 Bars
Monophonic Melodies dataset by ear. It has two phases. Phase 1 materializes
20 test melodies as WAV files with reference note tables. Phase 2 parses the
hand-written note tables and scores them against the references. The result
is the reference point for machine approaches.

Dataset melodies are 4 bars of 4/4: 16 quarter-note beats, quantized to a
0.25-beat grid. Audio is rendered at 120 BPM, so one beat is 0.5 seconds.

## Materialize

Materialization needs a General MIDI SoundFont. Point `SCRIBR_SOUNDFONT` at
one first:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.human_baseline.materialize
```

`--soundfont PATH` overrides the environment variable.

The command samples 20 examples from the first 1,000 test records with seed 42
and assigns the four instruments round robin, 5 examples each. It writes:

```text
artifacts/
├── manifest.json
├── reference/<id>.txt
└── wav/<id>.wav
```

Running it again reuses the existing manifest; `--force` rebuilds
everything. The manifest records the config, the sampled candidate indices,
and file hashes. It holds no timestamps or filesystem paths, so the same
inputs always produce the same file. The WAV directory is gitignored; the
references, manifest, and transcriptions are committed.

## Transcribe

To create the human transcription, the human should listen to each file in
`artifacts/wav/` and record the notes they hear in
`transcriptions/<id>.txt`, using the ID from the manifest (for example
`000-piano.txt`).

Each transcription should use one note per row in the following format:

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
optional. The human does not have to align times to the 0.25-beat grid.
Scoring uses the times exactly as written and reports how many rows fall off
the grid.

The human should not open `artifacts/reference/`. Those note tables are the
ground truth, and reading them would defeat the experiment.

The instrument release tail should be ignored. A struck or bowed note may
continue sounding after its notated offset, but a fading tail should not be
transcribed as a new note. The transcription should represent the attacks
that are heard, not the decay.

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
