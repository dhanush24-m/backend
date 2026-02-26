"""
Application entry point.
Wires together all components and registers routes.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.adapters import build_llm, build_stt, build_tts
from app.config import get_settings
from app.core.concurrency import ConcurrencyController
from app.core.logging import configure_logging, get_logger
from app.core.rate_limiter import RateLimiter
from app.gateway.websocket_handler import WebSocketHandler
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.session_manager import SessionManager

# ── Bootstrap logging before anything else ────────────────────────────────────
configure_logging()
logger = get_logger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────────
settings = get_settings()

session_manager = SessionManager()
concurrency_controller = ConcurrencyController(
    max_concurrent=settings.max_concurrent_pipelines,
    acquire_timeout=2.0,
)
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)
orchestrator = PipelineOrchestrator(
    stt=build_stt(),
    llm=build_llm(),
    tts=build_tts(),
)
ws_handler = WebSocketHandler(
    session_manager=session_manager,
    orchestrator=orchestrator,
    concurrency_controller=concurrency_controller,
    rate_limiter=rate_limiter,
)


# ── Application lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    session_manager.start_background_cleanup()
    logger.info(
        "Voice AI backend started",
        extra={
            "max_concurrent_pipelines": settings.max_concurrent_pipelines,
            "rate_limit": f"{settings.rate_limit_requests}/{settings.rate_limit_window_seconds}s",
        },
    )
    yield
    await session_manager.stop()
    logger.info("Voice AI backend shut down")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice AI Backend",
    description="WebSocket-based voice-to-voice AI system (STT → LLM → TTS)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await ws_handler.handle(websocket)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "active_sessions": len(session_manager._sessions),
        "pipeline_slots_available": concurrency_controller.available_slots,
        "concurrency_stats": {
            "acquired": concurrency_controller.stats.total_acquired,
            "rejected": concurrency_controller.stats.total_rejected,
            "current_active": concurrency_controller.stats.current_active,
        },
    }


@app.get("/")
async def root():
    return {"message": "Voice AI Backend running. Connect via WebSocket at /ws/voice"}
