"""Gemini audio transcription with bounded requests and response caching."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx2
import numpy as np
from google import genai
from google.genai import errors, types

from scribr.representation import Melody
from scribr.wav import encode_wav

from .table import PROMPT_VERSION, parse_response, transcription_prompt

# Leave room for base64 and request metadata below the inline request limit.
_MAX_WAV_BYTES = 10 * 1024 * 1024
_RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


class LLMTranscriptionError(RuntimeError):
    """A request failed or did not yield a valid, complete note table."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Gemini settings, excluding credentials and local cache paths.

    Attributes:
        model: Audio-capable Gemini model ID, defaulting to
            `SCRIBR_LLM_MODEL`. Required for non-silent audio.
        quantize: Round predictions to 0.25 beats; `False` keeps raw times.
        max_attempts: Total attempts per clip, including format repairs.
        timeout_seconds: Per-request timeout, not an overall run deadline.
        min_interval_seconds: Minimum interval between API request starts.
        retry_delay_seconds: Initial exponential backoff, capped at 60 s.
        max_output_tokens: Output budget; truncated responses are rejected.
    """

    model: str | None = None
    quantize: bool = True
    max_attempts: int = 3
    timeout_seconds: float = 60.0
    min_interval_seconds: float = 1.0
    retry_delay_seconds: float = 2.0
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        model = (
            self.model
            if self.model is not None
            else os.getenv("SCRIBR_LLM_MODEL")
        )
        if model is not None and (
            not isinstance(model, str) or not model.strip()
        ):
            raise ValueError("model must be a non-empty Gemini model ID")
        object.__setattr__(self, "model", model.strip() if model else None)
        if not isinstance(self.quantize, bool):
            raise ValueError("quantize must be a boolean")
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
    """Implement `Transcriber` using inline WAV audio and Gemini note tables.

    The client is created lazily from `GEMINI_API_KEY`. Silence and valid
    cached responses need no credentials. Failures raise an error rather
    than becoming empty predictions that would distort evaluation scores.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        cache_dir: str | Path | None = None,
        client: genai.Client | None = None,
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
        if len(wav) > _MAX_WAV_BYTES:
            raise ValueError(
                "audio exceeds the 10 MiB inline WAV limit; use shorter clips"
            )
        if not np.any(waveform):
            return Melody()
        if not self.config.model:
            raise ValueError(
                "set SCRIBR_LLM_MODEL to an audio-capable Gemini model ID"
            )

        duration = waveform.size / sample_rate * tempo / 60
        prompt = transcription_prompt(tempo, duration)
        cache_path = self._cache_path(wav, prompt)
        cached = self._read_cache(cache_path, duration)
        if cached is not None:
            self.cache_hits += 1
            return cached

        client = self._get_client()
        repair = ""
        for attempt in range(self.config.max_attempts):
            try:
                self._wait_for_rate_limit()
                self.api_calls += 1
                response = client.models.generate_content(
                    model=self.config.model,
                    contents=[
                        prompt + repair,
                        types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        candidate_count=1,
                        max_output_tokens=self.config.max_output_tokens,
                        response_mime_type="text/plain",
                    ),
                )
                text = self._response_text(response)
                melody = parse_response(
                    text, duration, quantize=self.config.quantize
                )
            except errors.APIError as error:
                # Provider payloads can contain request data; omit them.
                if error.code not in _RETRYABLE_CODES:
                    raise LLMTranscriptionError(
                        f"Gemini rejected the request (HTTP {error.code}); "
                        "check key and model access"
                    ) from None
                failure = f"Gemini request failed (HTTP {error.code})"
            except (httpx2.TransportError, TimeoutError, ConnectionError):
                failure = "Gemini request timed out or lost its connection"
            except ValueError:
                failure = "Gemini returned an invalid or incomplete note table"
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

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "set GEMINI_API_KEY before making uncached requests"
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
            "provider": "gemini",
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "audio_sha256": hashlib.sha256(wav).hexdigest(),
            "max_output_tokens": self.config.max_output_tokens,
            "temperature": 0,
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        # Quantization is local postprocessing; raw tables can serve both arms.
        return self.cache_dir / f"{digest}.txt"

    def _read_cache(self, path: Path | None, duration: float) -> Melody | None:
        if path is None or not path.is_file():
            return None
        try:
            return parse_response(
                path.read_text(encoding="utf-8"),
                duration,
                quantize=self.config.quantize,
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
