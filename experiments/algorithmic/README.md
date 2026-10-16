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
7. Snap onsets and offsets to the nearest 0.25 quarter-note beat by default.
   Half-grid ties round forward. Drop notes whose boundaries collapse to one
   grid point, and cap endings at the last grid point within the waveform.
   Monotone rounding preserves the ordering and non-overlap of notes.

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

## Evaluate the test baseline

The default `config.json` selects the first 20 test melodies with piano,
acoustic guitar, electric guitar, and violin assigned round robin, matching
human-baseline selection and synthesis settings. Use the same SoundFont.

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.algorithmic.evaluate
```

Download the test split first if needed using the main README's setup
instructions. `--soundfont PATH` overrides the environment variable.
`--config PATH` selects another shared `ExperimentSpec` JSON configuration.

The command materializes the selected examples into
`experiments/algorithmic/artifacts/` (`manifest.json`, `wav/`, and
`reference/`). Later runs reuse the manifest and read the same mono 16-bit
PCM WAVs that scoring hash-checks. Because `AlgorithmicConfig` is separate
from `ExperimentSpec`, changing `--parameters` reuses the manifest instead
of rendering again; changing `--config` raises until you delete the stale
`experiments/algorithmic/artifacts/` directory. `--artifacts-dir PATH`
writes the materialized artifacts and reports somewhere else. The
artifacts are gitignored.

## Tune on training audio

`tuning_config.json` selects the first 1,000 training melodies. This bounded
starting subset is larger than the test baseline; increase `sample_size` for
more extensive tuning as time permits. Do not choose parameters on test data.

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
  --config experiments/algorithmic/tuning_config.json \
  --parameters parameters.json \
  --artifacts-dir experiments/algorithmic/artifacts/tuning
```

Download the train split before tuning. Record each run's parameters and
inspect false/missing notes by instrument. Freeze parameters before running
the test comparison. Tuning is a comparison of measured runs, not an
automatic parameter search.

## Compare against the human baseline

Use the exact WAV files and references recorded in Ricardo's manifest:

```bash
uv run python -m experiments.algorithmic.evaluate \
  --manifest experiments/human_baseline/artifacts/manifest.json
```

If the files have not been materialized, run the existing human-baseline
materialization command with the agreed SoundFont first. This command reads
mono 16-bit PCM WAVs and checks their hashes and reference hashes against the
manifest. The materialized data needs no dataset or SoundFont at evaluation
time. Passing `--parameters PATH` uses your frozen validation settings.

The run writes `report.json` and a matching `report.md` into the artifacts
directory (`experiments/algorithmic/artifacts/` by default, or `--artifacts-dir
PATH`). The report contains the shared metadata block, overall and
per-instrument `mir_eval` metrics, per-example scores, metric tolerances, the
`AlgorithmicConfig` parameters, package versions, and the source manifest
provenance. It is written only after the complete run succeeds. Generated
reports are gitignored; choose explicitly which results to share.

Compare `F-measure_no_offset` first (pitch and onset together), and report
`F-measure` second (pitch, onset, and offset). Onset-only F1 is insufficient.
Silent examples remain included and score zero, matching the shared evaluator.

## Quantization ablation

`ablation.py` runs the transcriber on the same audio twice, once with
`quantize=false` and once with the quantizing default, and reports the
paired comparison. Use it to check whether snapping to the grid changes
scores on a split before trusting it:

```bash
uv run python -m experiments.algorithmic.ablation \
  --manifest experiments/human_baseline/artifacts/manifest.json
```

`--config PATH`, `--soundfont PATH`, and `--parameters PATH` behave as in
`evaluate.py`. `--parameters` sets the config shared by both arms and the
arms override `quantize`. The run writes `ablation.json` and `ablation.md`
to `--artifacts-dir` with overall scores, per-instrument means, per-example
deltas, and off-grid counts for each arm.

## Method references

- [librosa onset detection](https://librosa.org/doc/latest/api/onset.html).
- [librosa pYIN](https://librosa.org/doc/latest/api/generated/librosa.pyin.html).
