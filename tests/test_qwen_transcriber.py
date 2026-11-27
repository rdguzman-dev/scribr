"""Offline Qwen streaming, audio transport, and retry checks."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx2
import numpy as np
import pytest

from scribr.llm import LLMConfig, LLMTranscriber, LLMTranscriptionError
from scribr.llm import qwen
from scribr.representation import Melody, Note
from scribr.transcription import Transcriber
from scribr.wav import encode_wav

_BASE = "https://ws-test.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/"
_AUDIO = np.full(200, 0.25, dtype=np.float32)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.setattr(qwen.time, "sleep", MagicMock())


def _config(**kwargs: object) -> LLMConfig:
    return LLMConfig(
        model="qwen3.8-omni-flash",
        provider="qwen",
        qwen_base_url=_BASE,
        min_interval_seconds=0,
        retry_delay_seconds=0,
        **kwargs,
    )


def _event(data: dict) -> str:
    return "data: " + json.dumps(data) + "\n\n"


def _answer(text: str = "A4, 1, 3", finish: str = "stop") -> httpx2.Response:
    body = _event(
        {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]}
    )
    body += _event({"choices": [{"delta": {"content": text[:4]}}]})
    body += _event(
        {
            "choices": [
                {"delta": {"content": text[4:]}, "finish_reason": finish}
            ]
        }
    )
    body += _event({"choices": [], "usage": {"total_tokens": 100}})
    body += "data: [DONE]\n\n"
    return httpx2.Response(
        200, text=body, headers={"content-type": "text/event-stream"}
    )


def test_streamed_qwen_audio_and_protocol(tmp_path: Path) -> None:
    received = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        received.append(request)
        return _answer()

    with httpx2.Client(
        base_url=_BASE, transport=httpx2.MockTransport(respond)
    ) as client:
        transcriber: Transcriber = LLMTranscriber(
            _config(), client=client, cache_dir=tmp_path
        )
        assert transcriber.transcribe(_AUDIO, 100, 120) == Melody(
            (Note(69, 1, 3),)
        )
        assert transcriber.transcribe(_AUDIO, 100, 120) == Melody(
            (Note(69, 1, 3),)
        )
        assert (transcriber.api_calls, transcriber.cache_hits) == (1, 1)
    assert len(received) == 1
    request = received[0]
    assert request.url.path == "/compatible-mode/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "qwen3.8-omni-flash"
    assert body["stream"] is True
    assert body["modalities"] == ["text"]
    prompt, audio = body["messages"][0]["content"]
    assert "120 BPM" in prompt["text"]
    encoded = audio["input_audio"]["data"]
    assert encoded.startswith("data:audio/wav;base64,")
    assert base64.b64decode(encoded.split(",", 1)[1]) == encode_wav(
        _AUDIO, 100
    )
    assert "A4, 1, 3" not in prompt["text"]
    (cache,) = tmp_path.glob("*.txt")
    assert cache.read_text() == "A4, 1, 3"
    assert "private reasoning" not in cache.read_text()


@pytest.mark.parametrize(
    "status,expected_calls", [(429, 2), (503, 2), (401, 1), (403, 1)]
)
def test_http_failure_handling(status: int, expected_calls: int) -> None:
    calls = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return (
            httpx2.Response(status, json={"error": "sensitive provider data"})
            if len(calls) == 1
            else _answer()
        )

    with httpx2.Client(
        base_url=_BASE, transport=httpx2.MockTransport(respond)
    ) as client:
        transcriber = LLMTranscriber(_config(max_attempts=2), client=client)
        if expected_calls == 1:
            with pytest.raises(LLMTranscriptionError) as failure:
                transcriber.transcribe(_AUDIO, 100, 120)
            assert "sensitive provider data" not in str(failure.value)
        else:
            assert len(transcriber.transcribe(_AUDIO, 100, 120)) == 1
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "body",
    [
        "data: [DONE]\n\n",
        _event({"choices": [{"delta": {"content": "A4, 1, 3"}}]}),
        _event(
            {
                "choices": [
                    {
                        "delta": {"content": "A4, 1, 3"},
                        "finish_reason": "length",
                    }
                ]
            }
        ),
        _event(
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "A4, 1, 3"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        "data: invalid JSON\n\n",
        _event({"choices": [{"delta": {"content": 42}}]}),
        _event(
            {
                "choices": [
                    {"delta": {"refusal": "refused"}, "finish_reason": "stop"}
                ]
            }
        ),
    ],
)
def test_incomplete_or_invalid_stream_never_becomes_prediction(
    body: str, tmp_path: Path
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text=body)

    with httpx2.Client(
        base_url=_BASE, transport=httpx2.MockTransport(respond)
    ) as client:
        transcriber = LLMTranscriber(
            _config(max_attempts=1), client=client, cache_dir=tmp_path
        )
        with pytest.raises(LLMTranscriptionError):
            transcriber.transcribe(_AUDIO, 100, 120)
    assert list(tmp_path.iterdir()) == []


def test_invalid_table_is_repaired_within_same_budget() -> None:
    received = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        received.append(json.loads(request.content))
        return _answer("not a table") if len(received) == 1 else _answer()

    with httpx2.Client(
        base_url=_BASE, transport=httpx2.MockTransport(respond)
    ) as client:
        assert (
            len(
                LLMTranscriber(
                    _config(max_attempts=2), client=client
                ).transcribe(_AUDIO, 100, 120)
            )
            == 1
        )
    assert (
        "previous response was invalid"
        in received[1]["messages"][0]["content"][0]["text"]
    )


def test_qwen_key_is_not_replaced_by_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "offline-gemini-key")
    with pytest.raises(ValueError, match="QWEN_API_KEY"):
        LLMTranscriber(_config()).transcribe(_AUDIO, 100, 120)


def test_qwen_key_and_endpoint_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scribr.llm import transcriber as implementation

    constructor = MagicMock()
    monkeypatch.setattr(implementation.httpx2, "Client", constructor)
    monkeypatch.setenv("QWEN_API_KEY", "offline-qwen-key")
    transcriber = LLMTranscriber(_config())
    transcriber._get_client()
    assert constructor.call_args.kwargs["headers"] == {
        "Authorization": "Bearer offline-qwen-key"
    }
    assert constructor.call_args.kwargs["follow_redirects"] is False
    transcriber.close()
    constructor.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/",
        "https://example.com/compatible-mode/v1",
        "https://key@dashscope-intl.aliyuncs.com/compatible-mode/v1",
        _BASE + "?key=secret",
    ],
)
def test_invalid_qwen_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match="QWEN_BASE_URL"):
        LLMConfig(provider="qwen", qwen_base_url=url)


def test_native_endpoint_normalizes_to_chat_endpoint() -> None:
    native = _BASE.replace("/compatible-mode/v1/", "/api/v1")
    assert (
        LLMConfig(provider="qwen", qwen_base_url=native).qwen_base_url == _BASE
    )


@pytest.mark.parametrize("noise", [0.0, 1 / 32768])
def test_leading_silence_origin_is_restored_on_requests_and_cache(
    tmp_path: Path,
    noise: float,
) -> None:
    received = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        received.append(json.loads(request.content))
        return _answer("A4, 0, 1")

    audio = np.concatenate((np.full(50, noise, np.float32), _AUDIO))
    with httpx2.Client(
        base_url=_BASE, transport=httpx2.MockTransport(respond)
    ) as client:
        transcriber = LLMTranscriber(
            _config(trim_leading_silence=True),
            client=client,
            cache_dir=tmp_path,
        )
        assert transcriber.transcribe(audio, 100, 120) == Melody(
            (Note(69, 1, 2),)
        )
        assert transcriber.transcribe(audio, 100, 120) == Melody(
            (Note(69, 1, 2),)
        )
    assert len(received) == 1
    parts = received[0]["messages"][0]["content"]
    assert "Timing check:" in parts[0]["text"]
    payload = parts[1]["input_audio"]["data"].split(",", 1)[1]
    assert base64.b64decode(payload) == encode_wav(_AUDIO, 100)


def test_time_origin_is_restored_before_quantization() -> None:
    from scribr.llm.table import parse_response

    assert parse_response("A4, 0.2, 0.7", 1, time_origin_beats=0.13).notes == (
        Note(69, 0.25, 0.75),
    )
    assert parse_response(
        "A4, 0.2, 0.7", 1, time_origin_beats=0.13, quantize=False
    ).notes == (Note(69, 0.33, 0.83),)


@pytest.mark.parametrize("threshold", [-1.0, 1.0, float("nan")])
def test_invalid_silence_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="silence_threshold"):
        _config(silence_threshold=threshold)


def test_provider_and_region_separate_cache_entries(tmp_path: Path) -> None:
    qwen_transcriber = LLMTranscriber(_config(), cache_dir=tmp_path)
    gemini = LLMTranscriber(LLMConfig(model="test-model"), cache_dir=tmp_path)
    assert qwen_transcriber._cache_path(
        b"audio", "prompt"
    ) != gemini._cache_path(b"audio", "prompt")
    another = LLMTranscriber(
        LLMConfig(
            model="qwen3.8-omni-flash",
            provider="qwen",
            qwen_base_url=_BASE.replace("ws-test", "ws-another"),
        ),
        cache_dir=tmp_path,
    )
    assert qwen_transcriber._cache_path(
        b"audio", "prompt"
    ) != another._cache_path(b"audio", "prompt")
