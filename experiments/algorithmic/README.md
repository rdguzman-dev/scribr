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

The default `config.json` selects the entire test split (`sample_size` is
`null`) with piano, acoustic guitar, electric guitar, and violin assigned
round robin. Use the same SoundFont.

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.algorithmic.evaluate
```

Download the test split first if needed using the main README's setup
instructions. The full split renders tens of thousands of WAV files, so the
first run is slow and writes a large `artifacts/` directory. `--soundfont
PATH` overrides the environment variable. `--config PATH` selects another
shared `ExperimentSpec` JSON configuration.

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

`tuning_config.json` selects the first 2,000 training melodies and
`validation_config.json` selects the first 2,000 validation melodies.
Selection is a prefix, so the tuning stages and the validation split stay
nested and reproducible. Do not choose parameters on test data.

Both sets are materialized fresh. About 2,000 examples is roughly 1.5 GB of
float32 audio in memory and 0.75 GB of WAV files on disk. The train prefix
was checked to be representative: the attribute means are stable across the
first 3,000 validation records.

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
the test comparison. Tuning is a comparison of measured runs; the search
below automates the parameter choice.

## Automatic tuning with Optuna

`tune.py` searches `AlgorithmicConfig` with Optuna. Each trial builds a
config, calls `AlgorithmicTranscriber.transcribe` end to end on a screen
subset, and returns the mean `F-measure_no_offset`. Nothing is cached or
scored on the test split.

The dataset-specific fields are fixed: `sample_rate=22050`, the dataset
pitch bounds, and `quantize=True`. Optuna suggests the rest:

| Parameter | Search |
| --- | --- |
| `frame_length` | 1024, 2048, 4096 |
| `hop_length` | 128, 256, 512 |
| `silence_db` | -60 to -20 |
| `min_note_seconds` | 0.02 to 0.15, log scale |
| `max_gap_seconds` | 0 to 0.08 |
| `onset_merge_seconds` | 0 to 0.1 |
| `attack_reject_ratio` | 0 to 1 |
| `attack_backtrack_seconds` | 0 to 0.1 |
| `rms_window_hops` | 1, 2, 4, 8 |

pYIN costs about 0.27 s per example, so scoring 100 trials on the full
2,000-example tuning set would spend roughly 15 hours in pYIN alone.
Only `frame_length` and `hop_length` feed pYIN, so the runner caches that
pass per example and reuses it across trials that share them. Later
trials over the same recording only rerun segmentation. On top of that,
the runner uses two stages. Stage 1 scores every trial on a 500-example
screen prefix; stage 2 re-scores the top 20 trials on the full tuning
set. The champion is the stage-2 best, not `study.best_trial`. This keeps
the search broad while making the final ranking less noisy.

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.algorithmic.tune \
  --config experiments/algorithmic/tuning_config.json \
  --validation-config experiments/algorithmic/validation_config.json \
  --trials 100 \
  --top-k 20 \
  --artifacts-dir experiments/algorithmic/artifacts/tuning
```

The champion is the best of many trials on a finite prefix, so its tuning
score is optimistic. `--validation-config` materializes a separate split
under `artifacts/tuning/validation/` and scores the champion there. Report
the validation score, not the tuning score.

Outputs land in the artifacts directory:

- `best_parameters.json`: the full champion config, accepted by
  `evaluate.py --parameters`.
- `trials_stage1.csv`: the Optuna trials table.
- `finalists_stage2.csv`: each finalist's full-set scores and config,
  ordered by stage-2 `F-measure_no_offset`.
- `validation_metrics.json`: held-out metrics when `--validation-config`
  is given.

Every stage-1 trial, every stage-2 finalist, and a summary run are logged
to the `algorithmic_tuning` MLflow experiment. The summary run carries the
champion parameters, the tuning and validation metrics, and the output
files. `--tracking-uri` points at another store. `--screen-examples`,
`--final-examples`, `--top-k`, `--seed`, and `--n-jobs` change the run; the
defaults assume the 2,000-example set. Leave `--n-jobs` at 1 because pYIN
is CPU-bound and parallel trials do not nest in MLflow.

## Compare against the human baseline

The human baseline covers a 40-example seeded, stratified random sample of
the test split, not the full split. Score the algorithmic transcriber on the
same examples with the exact WAV files and references recorded in the
manifest:

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
