"""Qwen's streamed audio-to-text HTTP interface."""

import base64
import json
import time

import httpx2


class QwenAPIError(RuntimeError):
    """Provider failure carrying only an HTTP code, not request data."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"Qwen request failed (HTTP {code})")


def timing_instructions(tempo: float, duration_beats: float) -> str:
    """Reinforce the distinction between seconds and quarter-note beats."""
    duration_seconds = duration_beats * 60 / tempo
    return f"""
Timing check: the attached file is {duration_seconds:.9g} seconds long,
which equals {duration_beats:.9g} quarter-note beats at {tempo:.9g} BPM.
One second equals {tempo / 60:.9g} beats. A timestamp of 0.5 seconds equals
{tempo / 120:.9g} beats; a timestamp of 1.0 seconds equals {tempo / 60:.9g} beats.
Use the first FILE SAMPLE as beat zero, including all initial silence.
Do not shift the first note to zero unless it starts at the first sample.
Do not shorten note durations by returning seconds instead of beats.
Check the timestamp conversion before returning the table."""


def generate_note_table(
    client: httpx2.Client,
    model: str,
    prompt: str,
    wav: bytes,
    *,
    max_output_tokens: int,
    timeout_seconds: float,
) -> str:
    """Send inline WAV audio and collect only the final answer's text.

    Qwen streams reasoning separately from the answer. Reasoning is never
    parsed as a transcription. A completed `stop` event is required so a
    disconnected or token-limited response cannot become a partial melody.

    Args:
        client: Authenticated HTTP client with the regional API base URL.
        model: Qwen audio-input model ID.
        prompt: Note-table instructions, without reference notes.
        wav: Complete mono PCM16 WAV file as bytes.
        max_output_tokens: Combined output budget for the request.
        timeout_seconds: Streaming deadline, also used as the read timeout.

    Returns:
        Completed response text for the shared note-table parser.

    Raises:
        QwenAPIError: If the provider rejects or fails the request.
        ValueError: If the stream is malformed, truncated, or empty.
        TimeoutError: If the streaming deadline is exceeded.
    """
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,"
                            + base64.b64encode(wav).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        "modalities": ["text"],
        "stream": True,
        "max_tokens": max_output_tokens,
    }
    started = time.monotonic()
    pieces: list[str] = []
    finish_reason = None
    with client.stream("POST", "chat/completions", json=body) as response:
        if not 200 <= response.status_code < 300:
            raise QwenAPIError(response.status_code)

        for line in response.iter_lines():
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("Qwen stream exceeded its deadline")
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise ValueError("invalid Qwen stream event")
            if event.get("error"):
                raise QwenAPIError(502)
            choices = event.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise ValueError("expected one Qwen candidate")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("invalid Qwen candidate")
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict) or delta.get("refusal"):
                raise ValueError("invalid or refused Qwen answer")
            text = delta.get("content")
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("Qwen answer must contain text")
                pieces.append(text)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    text = "".join(pieces)
    if finish_reason != "stop" or not text.strip():
        raise ValueError("Qwen answer was incomplete, blocked, or empty")
    return text
