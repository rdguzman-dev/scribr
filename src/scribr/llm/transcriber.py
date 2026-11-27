"""Gemini and Qwen audio transcription with bounded requests and caching."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx2
import numpy as np
from google import genai
from google.genai import errors, types

from scribr.representation import Melody
from scribr.wav import encode_wav

from .table import PROMPT_VERSION, parse_response, transcription_prompt
from .qwen import QwenAPIError, generate_note_table, timing_instructions

# Leave room for base64 and request metadata below the inline request limit.
_MAX_WAV_BYTES = 10 * 1024 * 1024
_MAX_QWEN_WAV_BYTES = 7 * 1024 * 1024
_RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


class LLMTranscriptionError(RuntimeError):
    """A request failed or did not yield a valid, complete note table."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Provider settings, excluding credentials and local cache paths.

    Attributes:
        model: Audio-capable model ID, defaulting to
            `LLM_MODEL`, falling back to `SCRIBR_LLM_MODEL`.
            Required for non-silent audio.
        quantize: Round predictions to 0.25 beats; `False` keeps raw times.
        max_attempts: Total attempts per clip, including format repairs.
        timeout_seconds: Per-request timeout, not an overall run deadline.
        min_interval_seconds: Minimum interval between API request starts.
        retry_delay_seconds: Initial exponential backoff, capped at 60 s.
        max_output_tokens: Output budget; truncated responses are rejected.
        provider: `gemini` or `qwen`; defaults to Gemini for compatibility.
        qwen_base_url: Regional Qwen endpoint, defaulting to `QWEN_BASE_URL`.
        trim_leading_silence: Crop leading samples below `silence_threshold`,
            then restore their duration in the returned note times.
        silence_threshold: Normalized amplitude floor for optional cropping.
            Two PCM16 steps exclude FluidSynth's low-level dither.
    """

    model: str | None = None
    quantize: bool = True
    max_attempts: int = 3
    timeout_seconds: float = 60.0
    min_interval_seconds: float = 1.0
    retry_delay_seconds: float = 2.0
    max_output_tokens: int = 8192
    provider: str = "gemini"
    qwen_base_url: str | None = None
    trim_leading_silence: bool = False
    silence_threshold: float = 2 / 32768

    def __post_init__(self) -> None:
        if self.provider not in ("gemini", "qwen"):
            raise ValueError("provider must be gemini or qwen")
        if self.provider == "qwen":
            base_url = self.qwen_base_url or os.getenv("QWEN_BASE_URL")
            if base_url is not None:
                parsed = urlsplit(base_url)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or not parsed.hostname.endswith(".aliyuncs.com")
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError(
                        "QWEN_BASE_URL must be an Alibaba HTTPS endpoint "
                        "without credentials or query parameters"
                    )
                # The console also shows a native API base; this adapter
                # uses Chat Completions on the same regional workspace.
                if parsed.path.rstrip("/") == "/api/v1":
                    parsed = parsed._replace(path="/compatible-mode/v1")
                if parsed.path.rstrip("/") != "/compatible-mode/v1":
                    raise ValueError(
                        "QWEN_BASE_URL must end with /compatible-mode/v1 or /api/v1"
                    )
                object.__setattr__(
                    self, "qwen_base_url", urlunsplit(parsed).rstrip("/") + "/"
                )
        model = (
            self.model
            if self.model is not None
            else os.getenv("LLM_MODEL", os.getenv("SCRIBR_LLM_MODEL"))
        )
        if model is not None and (
            not isinstance(model, str) or not model.strip()
        ):
            raise ValueError("model must be a non-empty model ID")
        object.__setattr__(self, "model", model.strip() if model else None)
        if model and model.startswith("qwen") and self.provider != "qwen":
            raise ValueError("Qwen model IDs require provider='qwen'")
        if model and model.startswith("gemini") and self.provider != "gemini":
            raise ValueError("Gemini model IDs require provider='gemini'")
        if not isinstance(self.quantize, bool):
            raise ValueError("quantize must be a boolean")
        if not isinstance(self.trim_leading_silence, bool):
            raise ValueError("trim_leading_silence must be a boolean")
        if (
            not math.isfinite(self.silence_threshold)
            or not 0 <= self.silence_threshold < 1
        ):
            raise ValueError("silence_threshold must be finite and in [0, 1)")
        for name in ("max_attempts", "max_output_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "timeout_seconds",
            "min_interval_seconds",
            "retry_delay_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.timeout_seconds == 0:
            raise ValueError("timeout_seconds must be positive")


class LLMTranscriber:
    """Implement `Transcriber` using inline WAV audio and LLM note tables.

    The client reads `LLM_API_KEY`, falling back to `GEMINI_API_KEY`.
    Qwen uses `QWEN_API_KEY` with `QWEN_BASE_URL`. Silence and valid
    cached responses need no credentials. Failures raise an error rather
    than becoming empty predictions that would distort evaluation scores.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        cache_dir: str | Path | None = None,
        client: genai.Client | httpx2.Client | None = None,
    ) -> None:
        """Configure transcription, optionally injecting an offline client.

        Args:
            config: Provider, timing, and request limits.
            cache_dir: Successful raw table cache; disable with `None`.
            client: Existing SDK client; ownership stays with its caller.
        """
        self.config = config or LLMConfig()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._client = client
        self._owns_client = False
        self._last_request: float | None = None
        self.api_calls = 0
        self.cache_hits = 0

    def close(self) -> None:
        """Release connections owned by this transcriber."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
            self._owns_client = False

    def transcribe(
        self, audio: np.ndarray, sample_rate: int, tempo: float
    ) -> Melody:
        """Return notes in quarter-note beats measured from the first sample.

        Args:
            audio: Mono floating-point synthesis waveform in `[-1, 1]`.
            sample_rate: Positive integer sample rate in Hz.
            tempo: Finite positive synthesis tempo in BPM.

        Returns:
            A validated, optionally quantized `Melody`.

        Raises:
            ValueError: If input audio, timing, or configuration is invalid.
            LLMTranscriptionError: If attempts fail or the API rejects input.
        """
        waveform = np.asarray(audio)
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate < 1
        ):
            raise ValueError("sample_rate must be a positive integer")
        if not math.isfinite(tempo) or tempo <= 0:
            raise ValueError("tempo must be finite and positive")
        wav = encode_wav(waveform, sample_rate)
        if np.any(np.abs(waveform) > 1):
            raise ValueError("audio must be in [-1, 1]")
        limit = (
            _MAX_QWEN_WAV_BYTES
            if self.config.provider == "qwen"
            else _MAX_WAV_BYTES
        )
        if len(wav) > limit:
            raise ValueError(
                "audio exceeds the provider's inline WAV limit; use shorter clips"
            )
        if not np.any(waveform):
            return Melody()
        if not self.config.model:
            raise ValueError(
                "set LLM_MODEL or SCRIBR_LLM_MODEL to an audio-capable "
                "model ID"
            )

        time_origin = 0.0
        if self.config.trim_leading_silence:
            active = np.abs(waveform) > self.config.silence_threshold
            if not np.any(active):
                return Melody()
            first_sample = int(np.argmax(active))
            time_origin = first_sample / sample_rate * tempo / 60
            waveform = waveform[first_sample:]
            wav = encode_wav(waveform, sample_rate)
        duration = waveform.size / sample_rate * tempo / 60
        prompt = transcription_prompt(tempo, duration)
        if self.config.provider == "qwen":
            prompt += timing_instructions(tempo, duration)
        cache_path = self._cache_path(wav, prompt)
        cached = self._read_cache(cache_path, duration, time_origin)
        if cached is not None:
            self.cache_hits += 1
            return cached

        client = self._get_client()
        provider_name = "Qwen" if self.config.provider == "qwen" else "Gemini"
        repair = ""
        for attempt in range(self.config.max_attempts):
            try:
                self._wait_for_rate_limit()
                self.api_calls += 1
                if self.config.provider == "qwen":
                    text = generate_note_table(
                        client,
                        self.config.model,
                        prompt + repair,
                        wav,
                        max_output_tokens=self.config.max_output_tokens,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                else:
                    response = client.models.generate_content(
                        model=self.config.model,
                        contents=[
                            prompt + repair,
                            types.Part.from_bytes(
                                data=wav, mime_type="audio/wav"
                            ),
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            max_output_tokens=self.config.max_output_tokens,
                            response_mime_type="text/plain",
                        ),
                    )
                    text = self._response_text(response)
                melody = parse_response(
                    text,
                    duration,
                    quantize=self.config.quantize,
                    time_origin_beats=time_origin,
                )
            except (errors.APIError, QwenAPIError) as error:
                # Provider payloads can contain request data; omit them.
                if error.code not in _RETRYABLE_CODES:
                    raise LLMTranscriptionError(
                        f"{provider_name} rejected the request (HTTP {error.code}); "
                        "check key and model access"
                    ) from None
                failure = f"{provider_name} request failed (HTTP {error.code})"
            except (httpx2.TransportError, TimeoutError, ConnectionError):
                failure = (
                    f"{provider_name} request timed out or lost its connection"
                )
            except ValueError:
                failure = (
                    f"{provider_name} returned an invalid or incomplete "
                    "note table"
                )
                # Keep raw responses and references out of the repair prompt.
                repair = (
                    "\nThe previous response was invalid. Return a complete "
                    "table following every rule above."
                )
            else:
                self._write_cache(cache_path, text)
                return melody

            if attempt + 1 < self.config.max_attempts:
                time.sleep(
                    min(60.0, self.config.retry_delay_seconds * 2**attempt)
                )

        raise LLMTranscriptionError(
            f"{failure} after {self.config.max_attempts} attempt(s); "
            "no prediction was scored"
        )

    def _get_client(self) -> genai.Client | httpx2.Client:
        if self._client is None:
            if self.config.provider == "qwen":
                api_key = os.getenv("QWEN_API_KEY")
                if not api_key or not api_key.strip():
                    raise ValueError(
                        "set QWEN_API_KEY before making uncached Qwen requests"
                    )
                if not self.config.qwen_base_url:
                    raise ValueError(
                        "set QWEN_BASE_URL to your regional Alibaba API endpoint"
                    )
                self._client = httpx2.Client(
                    base_url=self.config.qwen_base_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=self.config.timeout_seconds,
                    follow_redirects=False,
                )
                self._owns_client = True
                return self._client
            api_key = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY"))
            if not api_key or not api_key.strip():
                raise ValueError(
                    "set LLM_API_KEY or GEMINI_API_KEY before making "
                    "uncached requests"
                )
            self._client = genai.Client(
                api_key=api_key,
                vertexai=False,
                http_options=types.HttpOptions(
                    timeout=math.ceil(self.config.timeout_seconds * 1000),
                    # Use one budget for transport and format repairs.
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._owns_client = True
        return self._client

    def _wait_for_rate_limit(self) -> None:
        if self._last_request is not None:
            remaining = self.config.min_interval_seconds - (
                time.monotonic() - self._last_request
            )
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()

    @staticmethod
    def _response_text(response: types.GenerateContentResponse) -> str:
        candidates = response.candidates or []
        if (
            len(candidates) != 1
            or candidates[0].finish_reason != types.FinishReason.STOP
        ):
            raise ValueError(
                "model response was blocked, absent, or truncated"
            )
        content = candidates[0].content
        parts = content.parts if content is not None else []
        text = "".join(
            part.text or "" for part in parts or [] if not part.thought
        )
        if not text.strip():
            raise ValueError("model did not return a note table")
        return text

    def _cache_path(self, wav: bytes, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        request = {
            "provider": self.config.provider,
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "audio_sha256": hashlib.sha256(wav).hexdigest(),
            "max_output_tokens": self.config.max_output_tokens,
            "temperature": 0 if self.config.provider == "gemini" else None,
        }
        if self.config.provider == "qwen":
            request["base_url"] = self.config.qwen_base_url
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        # Quantization is local postprocessing; raw tables can serve both arms.
        return self.cache_dir / f"{digest}.txt"

    def _read_cache(
        self, path: Path | None, duration: float, time_origin: float = 0.0
    ) -> Melody | None:
        if path is None or not path.is_file():
            return None
        try:
            return parse_response(
                path.read_text(encoding="utf-8"),
                duration,
                quantize=self.config.quantize,
                time_origin_beats=time_origin,
            )
        except (UnicodeError, ValueError):
            # Interrupted or edited cache entries must not become silence.
            return None

    @staticmethod
    def _write_cache(path: Path | None, text: str) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
