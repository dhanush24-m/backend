"""
TTS Adapter — OpenAI TTS
========================
Chosen because:
  - tts-1 is fast (~0.5–1 s) and returns audio as a raw binary stream.
  - Multiple voices available; voice is configurable via environment.
  - Returns MP3 by default, which WebSocket clients can play directly.

To swap providers: extend TTSAdapter (e.g. ElevenLabs, Google Cloud TTS).
"""

from typing import Optional

from openai import AsyncOpenAI

from app.adapters.base import TTSAdapter
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAITTS(TTSAdapter):
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.tts_model
        self._voice = settings.tts_voice
        self._response_format = settings.tts_response_format

    async def synthesize(self, text: str) -> bytes:
        logger.debug("Sending text to TTS", extra={"chars": len(text)})

        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,  # type: ignore[arg-type]
            input=text,
            response_format=self._response_format,  # type: ignore[arg-type]
        )

        audio_bytes = response.content
        logger.debug("TTS audio received", extra={"bytes": len(audio_bytes)})
        return audio_bytes
