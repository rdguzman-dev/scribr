# Gemini and Qwen transcription

`scribr.llm.LLMTranscriber` implements the unchanged `Transcriber` protocol.
It receives synthesized mono floating-point audio, sample rate, and tempo.
`scribr.wav.encode_wav` encodes mono PCM16 WAV bytes in memory; the SDK sends
these bytes inline without a temporary file or a Files API upload.
Gemini remains the default provider. Qwen uses the same WAV encoder,
base prompt, note-table parser, canonical `Melody`, and evaluation metrics.

## Setup

Run `uv sync`. Set `LLM_API_KEY` in your terminal environment to your
Gemini Developer API key, and `LLM_MODEL` to an audio-input-capable Gemini
model ID available to that key. `GEMINI_API_KEY` and `SCRIBR_LLM_MODEL` remain
supported fallbacks; the `LLM_*` names take precedence when present.
There is no hardcoded model default. `--model` overrides either environment
variable. Keys are read only from the environment and never placed in
configs, prompts, reports, or caches.
Setting variables in one terminal does not set them in other terminals.
`.env` files are ignored by Git but are not automatically loaded.
To load a local `.env` containing `LLM_API_KEY`, `LLM_MODEL`, and optionally
`SCRIBR_SOUNDFONT`, use uv's explicit environment-file option:

```bash
uv run --env-file .env python -m experiments.llm.evaluate
```

Use this same prefix with the evaluation options below. Do not commit `.env`.

### Qwen3.8-Omni-Flash

For Qwen, configure `QWEN_API_KEY` and `QWEN_BASE_URL` in the environment
or your ignored `.env`. Keep Gemini credentials separate. The base URL
must match the key's region and workspace, for example
`https://WORKSPACE.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`.
A console URL ending in `/api/v1` is normalized to `/compatible-mode/v1`
on the same host. Credentials or query parameters in the URL are rejected.

```bash
uv run --env-file .env python -m experiments.llm.evaluate \
  --provider qwen --model qwen3.8-omni-flash \
  --manifest experiments/llm/artifacts/manifest.json \
  --max-examples 4 --max-attempts 1 \
  --trim-leading-silence \
  --artifacts-dir experiments/llm/artifacts/qwen-pilot
```

`--base-url` overrides `QWEN_BASE_URL`. Qwen requests use the existing
`httpx2` dependency and Alibaba's streamed Chat Completions endpoint.
The request sends inline base64 WAV audio and requests text output only.
Model thinking settings are left at the provider default; reasoning chunks
are ignored. Only a completed final answer is parsed and cached. A missing
completion marker, output-limit finish, refusal, or malformed stream fails
within the same bounded retry budget used by Gemini. Qwen uses a 7 MiB WAV
limit to leave room for base64 under the provider's 10 MB encoded-data limit.
Its cache includes provider, regional endpoint, model, and request settings,
so a Gemini result is never reused for Qwen.

The Qwen prompt adds the file duration in seconds and explicit examples of
seconds-to-beats conversion. This helps avoid reporting seconds in a beat
column. It supplies no reference pitches or event times.

`--trim-leading-silence` is optional and off by default. It crops leading
samples below a small amplitude floor, records their duration, and adds it back to
every returned onset and offset before quantizing onto the original grid.
The default `--silence-threshold` is `2/32768` in normalized amplitude,
about -84 dBFS, to exclude FluidSynth's low-level dither. Set it to `0` to
crop exactly zero samples only. Very quiet sounds below a nonzero floor
are treated as silence, so the threshold is explicit and reported. No
reference notes enter this processing. Cached tables remain relative to the supplied cropped audio; the
original time origin is restored on cache hits too. This preprocessing is
recorded in report parameters. It addresses models that reset time zero to
the first sounding note. It does not correct other timing or pitch errors.

Qwen credentials are read only from `QWEN_API_KEY`; the adapter does not
fall back to the Gemini key. Model Studio quotas and billing apply to live
requests. The unit tests mock the HTTP stream and make no paid calls.

The official `google-genai` dependency handles inline audio requests and
provider responses. Its default HTTP transport, `httpx2`, is an explicit
dependency so connection failures and timeouts use the same bounded retry
handling as HTTP errors. SDK retries are disabled so `max_attempts` is the total
application request budget per clip. Requests use a configurable timeout,
a minimum interval, and exponential backoff for transient HTTP failures.
Invalid or incomplete tables also consume this same retry budget.
Authentication and other permanent HTTP errors fail immediately.

## Audio to melody

1. Validate mono finite floating-point audio in `[-1, 1]`, sample rate, and
   tempo. Empty or exactly zero audio returns an empty `Melody` locally.
2. Encode WAV bytes and build a prompt containing tempo, duration, and
   notation rules. No ground-truth notes, reference files, or dataset
   attributes are supplied to Gemini.
3. Request only a CSV-like table, for example:

   ```text
   pitch, onset, offset
   C4, 0.25, 0.75
   F#4, 1.00, 1.50
   ```

4. Reject absent, blocked, or truncated responses. Use the existing
   `melody_from_text` parser, then enforce dataset pitches A0-C8, ordered
   non-overlapping notes, and offsets within the audio duration. Header-only
   output means silence; an empty response is an error. One Markdown fence
   is tolerated, but explanatory prose and malformed rows are rejected.
5. Round to 0.25-beat boundaries by default, rounding half steps upward.
   Bound offsets to the last grid point inside the clip and drop notes
   collapsed by rounding. `--unquantized` preserves validated raw timings.
   Leading rests, gaps, and repeated pitches stay separate. Returned times
   are quarter-note beats from the first audio sample, never seconds or MIDI.

The prompt asks for physical note releases rather than instrument tails.
Parsing cannot verify musical correctness; a valid table can still be an
inaccurate transcription. Actual quality requires a live evaluation.

## Evaluation

The default config selects four test examples, one per instrument in manifest
order. It uses the same synthesis settings as the algorithmic baseline.
For prompt development, use a separate config with the `validation` split;
keep the test split for the final comparison.

With the dataset, FluidSynth, and `SCRIBR_SOUNDFONT` configured:

```bash
uv run python -m experiments.llm.evaluate
```

This preserves the common persistent `wav/`, `reference/`, and `manifest.json`
workflow. Each file is verified against its manifest hash before decoding.
To compare against the exact human/algorithmic baseline recordings instead:

```bash
uv run python -m experiments.llm.evaluate \
  --manifest experiments/human_baseline/artifacts/manifest.json
```

The manifest command requires those existing WAVs on disk, but no dataset or
SoundFont to resynthesize them. Do not substitute other recordings for missing
files. Both modes default to at most **4 examples**, even if the source config
or manifest is larger. `--max-examples 40` explicitly raises that cap.
For a config, the cap is applied before synthesis; changing materialization
settings requires a fresh `--artifacts-dir` to preserve manifest provenance.

Useful controls:

```bash
uv run python -m experiments.llm.evaluate --unquantized
uv run python -m experiments.llm.evaluate --max-examples 8 --max-attempts 2 \
  --min-interval-seconds 2 --timeout-seconds 90 \
  --artifacts-dir experiments/llm/artifacts/eight
uv run python -m experiments.llm.evaluate --no-cache
```

By default, successful raw tables are cached under
`experiments/llm/artifacts/responses/`. `--cache-dir` changes this location;
`--no-cache` disables reuse and writing. For a custom location outside the
ignored artifacts tree, use the ignored `.scribr-llm-cache/` directory.
Cache keys cover encoded audio, tempo/duration prompt, model, prompt version,
and generation settings. Quantization is local, so both timing modes reuse
the same raw table without another request. Invalid cache entries are ignored.
Valid cache hits work without an API key, but still require the same model ID.
Successful clips are cached immediately, so they survive a later clip failing.
Caching does not coordinate simultaneous processes or prevent charges for
failed requests. Four uncached, non-silent clips with three attempts allow
at most twelve API requests in one sequential run.

Errors stop evaluation; they are not scored as silence or omitted from a mean.
Reports are written only after every selected example succeeds. A failed run
can leave reports from an earlier run in the artifacts directory; only a
successful run's printed report paths identify new results.

Outputs are `predictions/*.txt`, `report.json`, and `report.md`, using common
`mir_eval` scoring and report utilities:

- Primary: **`F-measure_no_offset`**, combined pitch-and-onset F1.
- Secondary: **`F-measure`**, pitch, onset, and offset F1.

Reports include the complete source manifest, selected IDs, request/cache
counts, model, quantization, prompt version, generation settings, package
versions, and per-instrument/per-example metrics. A four-example pilot is
not a complete evaluation of the split. Compare approaches on identical IDs.

## Offline tests

```bash
uv run pytest tests/test_llm_transcriber.py tests/test_llm_experiment.py tests/test_wav_writer.py
uv run pytest
```

All LLM requests in tests are mocked. No API key, dataset download, or paid
requests are needed. The existing optional FluidSynth and Basic Pitch
artifact checks retain their normal skip conditions.
