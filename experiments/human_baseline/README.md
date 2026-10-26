# Human baseline

This experiment measures how well a person can transcribe the 4 Bars
Monophonic Melodies dataset by ear. It has two phases. Phase 1 materializes a
40-example human evaluation subset as WAV files with reference note tables.
Phase 2 parses the hand-written note tables and scores them against the
references.

## Materialize

Materialization needs a General MIDI SoundFont. Point `SCRIBR_SOUNDFONT` at
one first:

```bash
export SCRIBR_SOUNDFONT=GeneralUser-GS.sf2
uv run python -m experiments.human_baseline.materialize
```

`--soundfont PATH` overrides the environment variable.

The command draws the 40-example subset with a seeded, stratified random
sample. `config.json` sets `sample_size` to 40 and `seed` to 42. Selection
covers the whole test split, and round-robin assignment of the four
instruments (piano, acoustic guitar, electric guitar, and violin) gives 10
examples per instrument. Example IDs use the dataset record index, for
example `1041-piano`, not the position in the sample. It writes:

```text
artifacts/
├── manifest.json
├── reference/<id>.txt
└── wav/<id>.wav
```

Running it again reuses the existing manifest; `--force` rebuilds
everything. The manifest records the config, the selected dataset indices,
and file hashes. It holds no timestamps or filesystem paths, so the same
inputs always produce the same files. The WAV directory is gitignored. The
references, manifest, transcriptions, and the completed run's reports are
committed.

## Transcribe

Each file in `artifacts/wav/` is transcribed manually by listening to the
audio and recording the notes heard in `transcriptions/<id>.txt`, using the ID
from the manifest (for example, `1041-piano.txt`).

Each transcription uses one note per row in the following format:

```text
<pitch>, <onset>, <offset>
```

For example:

```text
C4, 0.0, 0.5
E4, 0.5, 1.0
G#4, 1.0, 2.0
```

Each row represents one note, with its pitch, onset, and offset. Pitches use
scientific notation (`C4` is MIDI 60), and sharps and flats both parse. Onset
and offset are quarter-note beats from the start of the WAV. The header row is
optional. Times do not have to be aligned to the 0.25-beat grid. Scoring uses
the times exactly as written and reports how many rows fall off the grid.
Whitespace is ignored.

Transcription uses GarageBand, which labels middle C as `C3`. The note
tables use scientific pitch notation to match Scribr's note tables, so they
are one octave above the labels GarageBand displays. Scoring uses the pitches
as written; no octave shift is applied.

### Transcription setup

Transcription is performed in GarageBand. A digital synthesizer capable of
emulating piano, acoustic guitar, electric guitar, and violin is available as
a reference instrument for identifying pitches by ear and using relative
pitch.

Ordinary audio-inspection tools are permitted. The waveform can be zoomed,
navigated, looped, and segmented into smaller regions to make difficult
passages easier to hear. The timing ruler can also be used to estimate note
onsets and offsets. These tools inspect and navigate the audio; they do not
automatically extract musical information.

Automatic transcription or note-extraction tools are not permitted. In
particular, automatic pitch detection, onset or offset detection, MIDI
transcription, and other tools that directly infer note identities or timings
from the audio are prohibited.

The reference note tables in `artifacts/reference/` must not be opened during
transcription. Those tables contain the ground truth, and reading them would
defeat the experiment.

The instrument release tail is ignored. A struck or bowed note may continue
sounding after its notated offset, but a fading tail is not transcribed as a
new note. The transcription represents the attacks that are heard, not the
decay.

### Timing log

The time spent on each transcription is recorded in `timing.csv`, one row
per example. The log is kept separate from the note tables in
`transcriptions/`, and the scoring pipeline does not read it.

Columns:

* `id`: example ID exactly as it appears in `transcriptions/` and the
  manifest (for example `1041-piano`).
* `datetime`: date and time the transcription was started, in
  `YYYY-MM-DD HH:MM` format.
* `duration_seconds`: how long the transcription took, recorded in seconds.
  Second-level precision is sufficient; small differences from the actual
  elapsed time are not meaningful.

An example row:

```text
1041-piano,2026-11-02 18:00,2520
```

## Score

Check progress at any time:

```bash
uv run python -m experiments.human_baseline.evaluate --check
```

`--check` prints the present, missing, and parse-error counts and exits 1 if
anything is missing or unparseable. It does not score.

Run the final evaluation when all 40 transcriptions exist:

```bash
uv run python -m experiments.human_baseline.evaluate
```

It writes `artifacts/report.json` and `artifacts/report.md` with overall and
per-instrument means, per-example metrics, off-grid counts, and environment
metadata. `F-measure_no_offset` is the headline metric because instrument
release tails make offset-aware scores noisy. To score an incomplete set
anyway, pass `--partial`; the report then lists the missing and unparseable
files instead of failing.

## Results

The 40-example random subset has not been transcribed yet, so there is no
current report. Run `--check` to see which note tables are outstanding.

### Pilot results (superseded)

The earlier 20-example prefix subset was replaced because its fixed prefix was
denser and more melodic than the full test set. Its transcriptions, timing log,
and report are kept under `pilot/` for reference. The pilot scores below are
unweighted means over examples, so each melody counts equally regardless of
note count. The run used the default tolerances, the pilot manifest's 120 BPM
tempo, and the `GeneralUser-GS.sf2` SoundFont.

| Metric | Score |
| --- | --- |
| `F-measure_no_offset` | 0.9478 |
| `Precision_no_offset` | 0.9492 |
| `Recall_no_offset` | 0.9465 |
| `Onset_F-measure` | 0.9875 |
| `F-measure` | 0.9152 |
| `Offset_F-measure` | 0.9603 |

| Instrument | N | `F-measure_no_offset` | `F-measure` |
| --- | --- | --- | --- |
| piano | 5 | 0.9852 | 0.9852 |
| acoustic guitar | 5 | 0.9724 | 0.8422 |
| electric guitar | 5 | 0.9359 | 0.9359 |
| violin | 5 | 0.8977 | 0.8977 |
| All | 20 | 0.9478 | 0.9152 |
