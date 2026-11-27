"""Offline tests for Gemini audio encoding, tables, retries, and caching."""

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import httpx2
from google.genai import Client as SDKClient
from google.genai import errors, types

from scribr.llm import LLMConfig, LLMTranscriber, LLMTranscriptionError
from scribr.llm import transcriber as implementation
from scribr.llm.table import parse_response, transcription_prompt
from scribr.representation import Melody, Note
from scribr.transcription import Transcriber

_TABLE = "pitch, onset, offset\nC4, 0.25, 0.75\nC4, 1, 1.5\n"
_AUDIO = np.full(200, 0.25, dtype=np.float32)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Prevent any test from creating a real provider client or sleeping."""
    constructor = MagicMock()
    monkeypatch.setattr(implementation.genai, "Client", constructor)
    monkeypatch.setattr(implementation.time, "sleep", MagicMock())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SCRIBR_LLM_MODEL", raising=False)
    return constructor


def _response(
    text: str = _TABLE, reason: types.FinishReason = types.FinishReason.STOP
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason=reason,
                content=types.Content(parts=[types.Part(text=text)]),
            )
        ]
    )


def _client() -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = _response()
    return client


def _config(**kwargs: object) -> LLMConfig:
    return replace(
        LLMConfig(model="test-audio-model", min_interval_seconds=0), **kwargs
    )


def test_protocol_and_inline_audio_without_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wave

    monkeypatch.chdir(tmp_path)
    client = _client()
    transcriber: Transcriber = LLMTranscriber(_config(), client=client)
    assert transcriber.transcribe(_AUDIO, 100, 120) == Melody(
        (
            Note(60, 0.25, 0.75),
            Note(60, 1, 1.5),
        )
    )
    request = client.models.generate_content.call_args.kwargs
    assert request["model"] == "test-audio-model"
    prompt, part = request["contents"]
    assert "120 BPM" in prompt and "4 beats" in prompt
    assert "C4, 0.25, 0.75" not in prompt
    assert part.inline_data.mime_type == "audio/wav"
    with wave.open(io.BytesIO(part.inline_data.data), "rb") as wav:
        assert (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
        ) == (100, 1, 2)
        assert (
            np.frombuffer(wav.readframes(200), "<i2").tolist() == [8192] * 200
        )
    assert request["config"].response_mime_type == "text/plain"
    assert list(tmp_path.iterdir()) == []


def test_prompt_preserves_beats_rests_and_no_reference() -> None:
    prompt = transcription_prompt(90, 12.75)
    assert "seconds * 90 / 60" in prompt
    assert "12.75" in prompt
    assert "FIRST AUDIO SAMPLE" in prompt
    assert "release tails" in prompt
    assert "reference" not in prompt.lower()


@pytest.mark.parametrize(
    "table", [_TABLE, "```csv\n" + _TABLE + "```", "c4, .25, .75\nC4, 1, 1.5"]
)
def test_table_uses_shared_pitch_names_and_keeps_repeated_notes(
    table: str,
) -> None:
    assert parse_response(table, 4).notes == (
        Note(60, 0.25, 0.75),
        Note(60, 1, 1.5),
    )
    assert parse_response("Bb3, 0, 1", 4).notes == (Note(58, 0, 1),)


def test_quantized_and_unquantized_keep_timing_origin() -> None:
    table = "C4, 0.38, 0.88\nD4, 1.01, 1.12\nE4, 1.5, 1.99"
    assert parse_response(table, 2).notes == (
        Note(60, 0.5, 1),
        Note(64, 1.5, 2),
    )
    assert parse_response(table, 2, quantize=False).notes == (
        Note(60, 0.38, 0.88),
        Note(62, 1.01, 1.12),
        Note(64, 1.5, 1.99),
    )
    assert parse_response("C4, 0.125, 0.9", 0.9).notes == (
        Note(60, 0.25, 0.75),
    )


@pytest.mark.parametrize(
    "table",
    [
        "",
        "  ",
        "```csv\nC4, 0, 1",
        "```csv\n```",
        "Here are your notes\nC4, 0, 1",
        '{"notes": []}',
        "60, 0, 1",
        "C4, 0",
        "C4, 0, 1, 2",
        "C4, nan, 1",
        "C4, 0, inf",
        "C4, -1, 1",
        "C4, 1, 1",
        "C4, 1, 0",
        "C4, 0, 4.01",
        "C4, 1, 2\nD4, 0, 1",
        "C4, 0, 2\nD4, 1, 3",
        "C0, 0, 1",
        "C9, 0, 1",
    ],
)
def test_invalid_tables_are_rejected(table: str) -> None:
    with pytest.raises(ValueError):
        parse_response(table, 4)


def test_silence_header_is_distinct_from_empty_response() -> None:
    assert parse_response("pitch, onset, offset", 4) == Melody()
    with pytest.raises(ValueError, match="empty"):
        parse_response("", 4)


@pytest.mark.parametrize(
    "audio", [np.zeros(0, np.float32), np.zeros(200, np.float32)]
)
def test_silent_audio_needs_no_model_key_or_client(
    audio: np.ndarray, offline: MagicMock
) -> None:
    assert LLMTranscriber().transcribe(audio, 100, 120) == Melody()
    offline.assert_not_called()


@pytest.mark.parametrize(
    "audio,rate,tempo",
    [
        (np.zeros((2, 10), np.float32), 100, 120),
        (np.zeros(10, np.int16), 100, 120),
        (np.array([np.nan], np.float32), 100, 120),
        (np.array([np.inf], np.float32), 100, 120),
        (np.array([1.01], np.float32), 100, 120),
        (_AUDIO, 0, 120),
        (_AUDIO, 1.5, 120),
        (_AUDIO, True, 120),
        (_AUDIO, 100, 0),
        (_AUDIO, 100, float("nan")),
    ],
)
def test_invalid_audio_never_calls_provider(
    audio: np.ndarray, rate: int, tempo: float
) -> None:
    client = _client()
    with pytest.raises(ValueError):
        LLMTranscriber(_config(), client=client).transcribe(audio, rate, tempo)
    client.models.generate_content.assert_not_called()


def test_inline_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation, "_MAX_WAV_BYTES", 50)
    with pytest.raises(ValueError, match="inline"):
        LLMTranscriber(_config(), client=_client()).transcribe(
            _AUDIO, 100, 120
        )


def test_model_and_key_are_required_for_live_requests() -> None:
    with pytest.raises(ValueError, match="SCRIBR_LLM_MODEL"):
        LLMTranscriber().transcribe(_AUDIO, 100, 120)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        LLMTranscriber(_config()).transcribe(_AUDIO, 100, 120)


def test_env_config_sdk_retry_budget_and_owned_client_close(
    monkeypatch: pytest.MonkeyPatch, offline: MagicMock
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("SCRIBR_LLM_MODEL", "env-model")
    offline.return_value = _client()
    transcriber = LLMTranscriber()
    assert transcriber.config.model == "env-model"
    transcriber.transcribe(_AUDIO, 100, 120)
    options = offline.call_args.kwargs
    assert options["api_key"] == "offline-test-key"
    assert options["vertexai"] is False
    assert options["http_options"].retry_options.attempts == 1
    assert options["http_options"].timeout == 60_000
    transcriber.close()
    offline.return_value.close.assert_called_once()
    transcriber.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": " "},
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_output_tokens": 0},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"min_interval_seconds": -1},
        {"retry_delay_seconds": -1},
        {"quantize": "false"},
    ],
)
def test_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _config(**kwargs)


def test_malformed_then_corrected_table_uses_one_retry_budget() -> None:
    client = _client()
    client.models.generate_content.side_effect = [
        _response("bad"),
        _response(),
    ]
    transcriber = LLMTranscriber(_config(), client=client)
    assert len(transcriber.transcribe(_AUDIO, 100, 120)) == 2
    assert transcriber.api_calls == 2
    assert (
        "previous response was invalid"
        in client.models.generate_content.call_args.kwargs["contents"][0]
    )


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", "gemini-4-hydrogen"])
def test_sdk_serializes_inline_audio_without_network(model: str) -> None:
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": _TABLE}],
                        },
                    }
                ],
            },
        )

    # Exercise real SDK serialization with an in-process HTTP transport.
    with httpx2.Client(transport=httpx2.MockTransport(respond)) as transport:
        with SDKClient(
            api_key="offline-key",
            vertexai=False,
            http_options=types.HttpOptions(httpx_client=transport),
        ) as sdk:
            result = LLMTranscriber(_config(model=model), client=sdk).transcribe(
                _AUDIO, 100, 120
            )
    assert len(result) == 2
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert (
        payload["contents"][0]["parts"][1]["inlineData"]["mime_type"]
        == "audio/wav"
    )
    assert payload["generationConfig"]["responseMimeType"] == "text/plain"
    # Gemini 3 and 4 do not support an explicit candidate count.
    assert "candidateCount" not in payload["generationConfig"]
    assert "offline-key" not in requests[0].content.decode()


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ReadTimeout("offline timeout"),
        httpx2.ConnectError("offline connection"),
    ],
)
def test_transport_failure_then_success(
    failure: httpx2.TransportError,
) -> None:
    client = _client()
    client.models.generate_content.side_effect = [failure, _response()]
    transcriber = LLMTranscriber(_config(), client=client)
    assert len(transcriber.transcribe(_AUDIO, 100, 120)) == 2
    assert transcriber.api_calls == 2


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_http_failure_then_success(code: int) -> None:
    client = _client()
    client.models.generate_content.side_effect = [
        errors.APIError(code, {}),
        _response(),
    ]
    transcriber = LLMTranscriber(_config(), client=client)
    assert len(transcriber.transcribe(_AUDIO, 100, 120)) == 2
    assert transcriber.api_calls == 2
    implementation.time.sleep.assert_called_once_with(2.0)


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_failure_does_not_retry_or_expose_payload(code: int) -> None:
    client = _client()
    client.models.generate_content.side_effect = errors.APIError(
        code, {"message": "secret-payload"}
    )
    with pytest.raises(LLMTranscriptionError, match=f"HTTP {code}") as failure:
        LLMTranscriber(_config(), client=client).transcribe(_AUDIO, 100, 120)
    assert "secret-payload" not in str(failure.value)
    assert client.models.generate_content.call_count == 1


@pytest.mark.parametrize(
    "response",
    [
        _response(""),
        _response(_TABLE, types.FinishReason.MAX_TOKENS),
        _response(_TABLE, types.FinishReason.SAFETY),
        types.GenerateContentResponse(),
        _response("not a table"),
    ],
)
def test_failed_responses_never_score_as_silence(
    response: types.GenerateContentResponse, tmp_path: Path
) -> None:
    client = _client()
    client.models.generate_content.return_value = response
    transcriber = LLMTranscriber(
        _config(max_attempts=2), client=client, cache_dir=tmp_path
    )
    with pytest.raises(LLMTranscriptionError, match="2 attempt"):
        transcriber.transcribe(_AUDIO, 100, 120)
    assert transcriber.api_calls == 2
    assert not list(tmp_path.iterdir())


def test_rate_limit_between_request_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        implementation.time, "monotonic", MagicMock(side_effect=[10, 10.5, 12])
    )
    transcriber = LLMTranscriber(
        _config(min_interval_seconds=2), client=_client()
    )
    transcriber.transcribe(_AUDIO, 100, 120)
    transcriber.transcribe(_AUDIO, 100, 120)
    implementation.time.sleep.assert_called_once_with(1.5)


def test_cache_replays_without_key_and_reuses_raw_unquantized_table(
    tmp_path: Path, offline: MagicMock
) -> None:
    client = _client()
    client.models.generate_content.return_value = _response("C4, .38, .88")
    first = LLMTranscriber(_config(), client=client, cache_dir=tmp_path)
    assert first.transcribe(_AUDIO, 100, 120).notes == (Note(60, 0.5, 1),)
    first.close()
    client.close.assert_not_called()
    second = LLMTranscriber(_config(quantize=False), cache_dir=tmp_path)
    assert second.transcribe(_AUDIO, 100, 120).notes == (Note(60, 0.38, 0.88),)
    assert (second.api_calls, second.cache_hits) == (0, 1)
    offline.assert_not_called()
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert all(
        "offline-test-key" not in p.read_text() for p in tmp_path.glob("*.txt")
    )


def test_model_audio_tempo_and_output_budget_separate_cache_entries(
    tmp_path: Path,
) -> None:
    client = _client()
    for config, audio, tempo in [
        (_config(), _AUDIO, 120),
        (_config(model="other-model"), _AUDIO, 120),
        (_config(), _AUDIO * 0.5, 120),
        (_config(), _AUDIO, 90),
        (_config(max_output_tokens=1024), _AUDIO, 120),
    ]:
        LLMTranscriber(config, client=client, cache_dir=tmp_path).transcribe(
            audio, 100, tempo
        )
    assert client.models.generate_content.call_count == 5
    assert len(list(tmp_path.glob("*.txt"))) == 5


def test_corrupted_cache_is_replaced_after_valid_response(
    tmp_path: Path,
) -> None:
    transcriber = LLMTranscriber(
        _config(), client=_client(), cache_dir=tmp_path
    )
    transcriber.transcribe(_AUDIO, 100, 120)
    (path,) = tmp_path.glob("*.txt")
    path.write_text("C4, nan, 1")
    assert len(transcriber.transcribe(_AUDIO, 100, 120)) == 2
    assert transcriber.api_calls == 2
    assert path.read_text() == _TABLE
