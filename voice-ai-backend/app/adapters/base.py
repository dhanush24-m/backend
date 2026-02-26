"""
Abstract base classes for AI provider adapters.
Any concrete adapter must implement these interfaces, making providers
fully replaceable without changing orchestration or handler code.
"""

from abc import ABC, abstractmethod
from typing import List


class STTAdapter(ABC):
    """Speech-to-Text: bytes → transcript string."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribe audio bytes to plain text.

        Parameters
        ----------
        audio_bytes : raw audio content (e.g. WebM, WAV, MP3)
        mime_type   : hint for the underlying API

        Returns
        -------
        Transcribed text string (may be empty if audio is silence/noise).
        """
        ...


class LLMAdapter(ABC):
    """Large Language Model: conversation history → assistant reply."""

    @abstractmethod
    async def chat(self, messages: List[dict]) -> str:
        """
        Run a chat completion.

        Parameters
        ----------
        messages : OpenAI-style message list
                   [{"role": "user"|"assistant"|"system", "content": "..."}]

        Returns
        -------
        Assistant reply as plain text.
        """
        ...


class TTSAdapter(ABC):
    """Text-to-Speech: text → audio bytes."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Parameters
        ----------
        text : input text to synthesize

        Returns
        -------
        Raw audio bytes (format depends on adapter configuration).
        """
        ...
