"""
LLM Adapter — OpenAI GPT
========================
Chosen because:
  - gpt-4o-mini provides strong quality at low cost and ~1–2 s latency.
  - Familiar chat-completion API; easy to swap for Claude, Gemini, etc.
  - Supports conversation history natively via the messages array.

To swap providers: extend LLMAdapter and update the factory.
"""

from typing import List, Optional

from openai import AsyncOpenAI

from app.adapters.base import LLMAdapter
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIGPTLLM(LLMAdapter):
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._system_prompt = settings.llm_system_prompt
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    async def chat(self, messages: List[dict]) -> str:
        # Prepend system message; downstream code only passes user/assistant turns
        full_messages = [{"role": "system", "content": self._system_prompt}] + messages

        logger.debug("Sending to LLM", extra={"turns": len(messages), "model": self._model})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,  # type: ignore[arg-type]
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        reply = response.choices[0].message.content or ""
        logger.debug("LLM reply received", extra={"chars": len(reply)})
        return reply.strip()
