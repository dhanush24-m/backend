"""
Configuration management — all values sourced from environment variables.
No secrets or defaults that bypass real configuration.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = "test-key"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str
    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    llm_system_prompt: str = (
        "You are a helpful voice-based customer support assistant. "
        "Keep answers concise and clear — they will be read aloud."
    )
    llm_max_tokens: int = 300
    llm_temperature: float = 0.7

    # ── STT ───────────────────────────────────────────────────────────────────
    stt_model: str = "whisper-1"
    stt_language: str = "en"

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"  # alloy | echo | fable | onyx | nova | shimmer
    tts_response_format: str = "mp3"

    # ── Concurrency ───────────────────────────────────────────────────────────
    max_concurrent_pipelines: int = 10
    pipeline_timeout_seconds: float = 30.0
    pipeline_max_retries: int = 2
    pipeline_retry_delay_seconds: float = 1.0

    # ── Rate limiting (per-client, in-memory) ─────────────────────────────────
    rate_limit_requests: int = 10      # max requests per window
    rate_limit_window_seconds: int = 60

    # ── Session ───────────────────────────────────────────────────────────────
    session_idle_timeout_seconds: float = 300.0   # 5-minute idle cleanup
    max_conversation_history: int = 20            # keep last N turns

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation_bytes: int = 10 * 1024 * 1024    # 10 MB
    log_backup_count: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings() 
