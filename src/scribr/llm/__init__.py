"""LLM-based monophonic transcription using Gemini or Qwen audio input."""

from .transcriber import LLMConfig, LLMTranscriber, LLMTranscriptionError

__all__ = ["LLMConfig", "LLMTranscriber", "LLMTranscriptionError"]
