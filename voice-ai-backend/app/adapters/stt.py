"""
STT Adapter — OpenAI Whisper
============================
Chosen because:
  - Whisper-1 is fast (~1–3 s for short clips), accurate, and cheap.
  - The API accepts raw binary audio in many common formats.
  - No special streaming setup required for the scope of this project.

To swap providers: create a new class that extends STTAdapter and
update the factory in adapters/__init__.py.
"""

import io
from typing import Optional

from openai import AsyncOpenAI

from app.adapters.base import STTAdapter
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIWhisperSTT(STTAdapter):
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.stt_model
        self._language = settings.stt_language

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        # Derive a sensible file extension from mime type for the API
        ext_map = {
            "audio/webm": "webm",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mp4": "mp4",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
        }
        ext = ext_map.get(mime_type, "webm")
        filename = f"audio.{ext}"

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        logger.debug("Sending audio to Whisper STT", extra={"bytes": len(audio_bytes)})

        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=(filename, audio_bytes, mime_type),
            language=self._language,
        )

        transcript = response.text.strip()
        logger.debug("Whisper transcript received", extra={"length": len(transcript)})
        return transcript
