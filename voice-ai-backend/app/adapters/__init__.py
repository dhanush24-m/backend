"""
Adapter factory.
Change the concrete classes here to swap AI providers globally.
"""

from app.adapters.base import LLMAdapter, STTAdapter, TTSAdapter
from app.adapters.llm import OpenAIGPTLLM
from app.adapters.stt import OpenAIWhisperSTT
from app.adapters.tts import OpenAITTS


def build_stt() -> STTAdapter:
    return OpenAIWhisperSTT()


def build_llm() -> LLMAdapter:
    return OpenAIGPTLLM()


def build_tts() -> TTSAdapter:
    return OpenAITTS()
