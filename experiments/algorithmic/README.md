# Algorithmic pYIN baseline

`src/scribr/algorithmic/transcriber.py` implements `Transcriber` without
training. It accepts only audio, sample rate, and the known synthesis tempo,
and returns canonical `Melody` notes. It never reads reference notes during
inference and does not use MIDI as an interface.

## How it works

1. Resample mono audio to the configured analysis rate (22,050 Hz by default).
2. Use `librosa.pyin` to estimate fundamental frequency and voicing. Round
   valid frequencies to MIDI pitches within the dataset range, 21–108.
3. Gate voiced frames using short-window RMS energy relative to the loudest
   frame. Bridge missing-pitch gaps up to 30 ms only when the pitch on both
   sides agrees. Remove brief pitch excursions surrounded by the same pitch.
4. Use `librosa.onset.onset_detect` for spectral attacks, reject candidates
   whose energy has fallen, and backtrack accepted attacks to preceding RMS
   minima. Stable pitch changes provide boundaries even without an attack.
   Attacks within a stable pitch region separate repeated notes.
5. End notes at the next pitch region, sustained unvoiced/quiet region, or
   waveform end. Clamp to the input duration and prevent overlapping notes.
   Discard regions shorter than 50 ms by default.
6. Convert seconds to quarter-note beats with `seconds * tempo / 60`.
   Keep detected times unquantized; do not snap to the reference grid.

Offsets are acoustic estimates, not recovered MIDI note-off events. A
SoundFont release can remain voiced after a reference note ends. The RMS gate
is relative to the whole recording, so very quiet notes can be lost, and
pYIN can make octave errors or miss short notes. These are baseline limitations
to measure across instruments, not guarantees of dataset accuracy.

## Install and check

From the repository root:

```bash
uv sync --locked
uv run pytest
```

Tests include real sine-wave transcription, input validation, rests, repeated
notes, pitch changes, short tracker dropouts, tempo conversion, resampling,
and verified manifest/report handling. They do not require a SoundFont.

## Tune on validation audio

The default `config.json` selects the first 20 validation melodies with
piano, acoustic guitar, electric guitar, and violin assigned round robin.
It uses the same rendering settings as the current human-baseline config.

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.algorithmic.evaluate
```

Download the validation split first if needed using the main README's setup
instructions. `--soundfont PATH` overrides the environment variable.
`--config PATH` selects another shared `ExperimentSpec` JSON configuration.

Tune `AlgorithmicConfig` parameters using a JSON mapping, for example:

```json
{
  "silence_db": -35.0,
  "max_gap_seconds": 0.02,
  "min_note_seconds": 0.05
}
```

```bash
uv run python -m experiments.algorithmic.evaluate \
  --parameters parameters.json \
  --output experiments/algorithmic/artifacts/validation.json
```

Keep tuning on validation data. Record each run's parameters and inspect
false/missing notes, especially by instrument. Freeze the parameters before
running the test comparison.

## Compare against the human baseline

Use the exact WAV files and references recorded in Ricardo's manifest:

```bash
uv run python -m experiments.algorithmic.evaluate \
  --manifest experiments/human_baseline/artifacts/manifest.json \
  --output experiments/algorithmic/artifacts/test.json
```

If the files have not been materialized, run the existing human-baseline
materialization command with the agreed SoundFont first. This command reads
mono 16-bit PCM WAVs and checks their hashes and reference hashes against the
manifest. The materialized data needs no dataset or SoundFont at evaluation
time. Passing `--parameters PATH` uses your frozen validation settings.

The JSON report contains all `mir_eval` metrics, per-example and per-instrument
scores, detected notes, metric tolerances, parameters, package versions, and
source provenance. It is written only after the complete run succeeds.
Generated reports are gitignored; choose explicitly which results to share.

Compare `F-measure_no_offset` first (pitch and onset together), and report
`F-measure` second (pitch, onset, and offset). Onset-only F1 is insufficient.
Silent examples remain included and score zero, matching the shared evaluator.

## Review and contribution

Explain the pYIN output, Hz-to-MIDI conversion, onset filtering, segmentation,
offset limitations, seconds-to-beats conversion, and metric selection before
submitting. Review every generated change and follow the course's AI citation
requirements once confirmed by the professor.

Branch: `feat/add-algorithmic-transcriber`.
Suggested Conventional Commit subject:

```text
feat(algorithmic): add pYIN transcription and evaluation
```

Open a pull request for Ricardo to review; the code is not considered reviewed
or merged merely because its tests pass.

## Method references

- [librosa onset detection](https://librosa.org/doc/latest/api/onset.html).
- [librosa pYIN](https://librosa.org/doc/latest/api/generated/librosa.pyin.html).
