# Voice-to-Voice AI Backend

A low-latency, concurrent, voice-to-voice customer support backend built with FastAPI WebSockets, following a modular, layered architecture.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd voice-ai-backend

# 2. Copy and fill in your API key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# 3. Run with Docker Compose
docker compose up --build

# The server is now available at:
#   WebSocket:  ws://localhost:8000/ws/voice
#   Health:     http://localhost:8000/health
```

### Run Tests (without Docker)

```bash
pip install -r requirements.txt
pytest
```

---

## Architecture

```
┌────────────────────────────────────────────────┐
│                 WebSocket Client                │
└───────────────────────┬────────────────────────┘
                        │  Binary audio chunks + empty sentinel
                        ▼
┌────────────────────────────────────────────────┐
│            Gateway Layer                        │
│  app/gateway/websocket_handler.py               │
│  • Accepts connection, assigns session          │
│  • Rate-limits per client (in-memory)           │
│  • Applies concurrency control                  │
│  • Delegates utterances to orchestrator         │
│  • Sends audio response + metadata to client    │
└───────────────────────┬────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
┌─────────────────────┐   ┌───────────────────────┐
│  Session Manager     │   │  Concurrency Ctrl      │
│  app/services/       │   │  app/core/concurrency  │
│  • Per-conn history  │   │  • asyncio.Semaphore   │
│  • Idle eviction     │   │  • Backpressure via    │
│  • Thread-safe lock  │   │    timeout+rejection   │
└─────────────────────┘   └───────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────┐
│           Pipeline Orchestrator                 │
│  app/pipeline/orchestrator.py                   │
│  • STT → LLM → TTS sequential execution        │
│  • Per-stage timeout + retry with back-off     │
│  • LatencyReport populated at each stage        │
└──────┬─────────────┬──────────────┬────────────┘
       │             │              │
       ▼             ▼              ▼
  STTAdapter     LLMAdapter     TTSAdapter
  (Whisper)       (GPT)          (OpenAI TTS)
  app/adapters/stt.py   llm.py   tts.py
```

### Layer Responsibilities

| Layer | File(s) | Responsibility |
|---|---|---|
| **Gateway** | `gateway/websocket_handler.py` | WebSocket lifecycle, protocol, rate limiting |
| **Orchestrator** | `pipeline/orchestrator.py` | Stage coordination, retries, timeouts, latency |
| **Session Manager** | `services/session_manager.py` | Conversation state, idle cleanup |
| **Adapters** | `adapters/{stt,llm,tts}.py` | External API calls, one provider per file |
| **Concurrency** | `core/concurrency.py` | Semaphore, slot management, backpressure |
| **Rate Limiter** | `core/rate_limiter.py` | Sliding-window per-client limiting |
| **Config** | `config.py` | All env-based settings, no defaults that skip config |
| **Logging** | `core/logging.py` | Structured JSON, rotating file + console |
| **Metrics** | `metrics/latency.py` | `measure()` context manager, LatencyReport |

---

## API Choices

### STT — OpenAI Whisper (`whisper-1`)
- Highly accurate, multi-language, accepts many audio formats directly.
- ~1–3 s latency for short clips; no streaming required at this scope.
- Cheapest accurate option in the OpenAI ecosystem.
- **To swap**: implement `STTAdapter` and change `adapters/__init__.py:build_stt()`.

### LLM — OpenAI GPT (`gpt-4o-mini`)
- Strong instruction-following, fast (~1–2 s), inexpensive.
- Natively supports conversation history via the messages array.
- **To swap**: implement `LLMAdapter` and change `build_llm()`.  
  (Anthropic Claude, Google Gemini, or a local model via Ollama would all work.)

### TTS — OpenAI TTS (`tts-1`)
- ~0.5–1 s latency, returns raw MP3 bytes — easy to relay over WebSocket.
- Six voices selectable via config.
- **To swap**: implement `TTSAdapter` and change `build_tts()`.

---

## WebSocket Protocol

Connect: `ws://host:8000/ws/voice`

**Client → Server**

| Frame | Meaning |
|---|---|
| Binary (non-empty) | Audio data chunk |
| Binary (empty, 0 bytes) | End-of-utterance — triggers pipeline |
| Text `{"cmd":"reset"}` | Clear conversation history |

**Server → Client**

| Frame | Meaning |
|---|---|
| Text `{"status":"processing"}` | Pipeline started |
| Binary | Response audio (MP3) |
| Text `{"status":"done","transcript":"...","latency":{...}}` | Pipeline complete |
| Text `{"status":"error","message":"..."}` | Non-fatal error |

---

## Concurrency Model

An `asyncio.Semaphore` with a configurable count (`MAX_CONCURRENT_PIPELINES`, default 10) limits simultaneous pipeline executions. The gateway acquires a slot before running the orchestrator and releases it on completion or error.

**Backpressure strategy**: if the semaphore cannot be acquired within 2 seconds, the client receives an error frame and must retry. This keeps latency predictable and prevents unbounded queue build-up.

**Why not a queue?** A queue hides latency from the client and can grow unboundedly. Rejection is explicit and observable; clients can implement their own retry logic.

**Multi-instance limitation**: the semaphore is in-process only. For horizontal scaling, a Redis-backed distributed counter or a dedicated queue (e.g. RabbitMQ, Redis Streams) would replace it.

---

## Rate Limiting

Implemented as a per-client sliding-window counter (`core/rate_limiter.py`):
- Default: 10 requests per 60-second window.
- Keyed by `{remote_ip}:{remote_port}` (client connection address).
- In-memory deque of timestamps; entries older than the window are purged on each check.

**Limitations**: resets on server restart; not shared across instances. In production, use Redis with a Lua script for atomicity.

---

## Latency Handling

Each pipeline stage is wrapped in an async context manager (`metrics/latency.py:measure()`) that records elapsed time to a `LatencyReport`. On completion the report is:
1. Logged as a structured JSON entry with per-stage and total milliseconds.
2. Sent to the client in the `done` status frame.

Stage-level logging also happens immediately on completion so you can correlate individual stage durations even in concurrent sessions.

---

## Logging Strategy

- **Format**: structured JSON; every log line is independently parseable.
- **Transport**: console stdout (for Docker/container log aggregators) + rotating file (`logs/app.log`, 10 MB / 5 backups).
- **Context**: `session_id` and `request_id` are injected via `contextvars` so all log lines within a request are correlated.
- **Security**: raw audio bytes and API keys are never logged; only byte counts are recorded.

Sample log entry:

```json
{
  "ts": "2026-02-23T10:00:01Z",
  "level": "INFO",
  "logger": "app.pipeline.orchestrator",
  "session_id": "abc-1234",
  "request_id": "e7f2a1b0",
  "msg": "Pipeline latency report",
  "latency_stt_ms": 1423.1,
  "latency_llm_ms": 1871.5,
  "latency_tts_ms": 612.4,
  "latency_total_ms": 3907.0
}
```

---

## Error Handling & Resilience

| Scenario | Behaviour |
|---|---|
| External API timeout | Stage retried up to `PIPELINE_MAX_RETRIES` times with exponential back-off; then `PipelineError` → client error frame |
| External API error | Same as timeout |
| Empty transcript (silence) | Immediate `PipelineError` at STT stage; no LLM/TTS calls made |
| Client disconnects mid-pipeline | WebSocket `disconnect` event exits the handler; session is cleaned up; in-flight pipeline completes but result is discarded |
| Server at capacity | Client receives error frame immediately; no pipeline started |
| Rate limit exceeded | Client receives error frame; audio buffer cleared |
| Unhandled exception | Top-level `except Exception` in handler logs and sends error frame; server continues |

Failures **never crash the server** — the top-level handler in `websocket_handler.py` ensures cleanup always runs via `finally`.

---

## Security Considerations

- **No raw audio logged** — only byte counts.
- **No secrets logged** — Pydantic settings load from env; logging configuration explicitly avoids config dumps.
- **Non-root Docker user** — the container runs as UID 1001.
- **CORS** — currently `allow_origins=["*"]`; should be restricted to known origins in production.
- **No authentication** — out of scope per the task; in production add JWT verification in the WebSocket handshake.
- **Input size** — audio buffer has no hard size cap currently; add one in production to prevent memory exhaustion.

---

## What Would Be Improved in Production

1. **Audio streaming** — pipe audio chunks directly to the STT API as they arrive instead of buffering the full utterance, cutting perceived latency.
2. **Distributed concurrency** — replace the in-process semaphore with Redis INCR/DECR for multi-instance deployments.
3. **Persistent session storage** — store conversation history in Redis with TTL to survive restarts.
4. **Authentication** — JWT or API-key verification in the WebSocket handshake before accepting any data.
5. **Input validation** — reject audio buffers above a maximum size and enforce expected MIME type.
6. **Metrics export** — expose Prometheus metrics (`/metrics`) for latency histograms, error rates, and active session counts.
7. **Separate worker service** — offload pipeline execution to a worker pool (e.g. via Redis Streams or a task queue) to decouple the WebSocket gateway from CPU/IO-heavy AI processing.
8. **TTS streaming** — OpenAI's TTS API supports streaming; returning audio chunks progressively would further reduce time-to-first-audio.

---

## Project Structure

```
voice-ai-backend/
├── app/
│   ├── main.py                   # FastAPI app, routes, lifespan hooks
│   ├── config.py                 # Environment-based configuration (Pydantic)
│   ├── gateway/
│   │   └── websocket_handler.py  # WebSocket lifecycle, protocol parsing
│   ├── pipeline/
│   │   └── orchestrator.py       # STT→LLM→TTS coordination, retries, timeouts
│   ├── adapters/
│   │   ├── base.py               # Abstract STTAdapter, LLMAdapter, TTSAdapter
│   │   ├── stt.py                # OpenAI Whisper implementation
│   │   ├── llm.py                # OpenAI GPT implementation
│   │   ├── tts.py                # OpenAI TTS implementation
│   │   └── __init__.py           # Factory functions (swap providers here)
│   ├── services/
│   │   └── session_manager.py    # Per-session history, idle eviction
│   ├── core/
│   │   ├── concurrency.py        # Semaphore-based pipeline limiting
│   │   ├── rate_limiter.py       # Sliding-window per-client rate limiting
│   │   └── logging.py            # JSON formatter, rotating file handler
│   └── metrics/
│       └── latency.py            # LatencyReport, measure() context manager
├── tests/
│   ├── test_orchestrator.py      # Full orchestrator behaviour (mocked APIs)
│   ├── test_session_manager.py   # Session CRUD, history, eviction
│   └── test_concurrency.py       # Semaphore limits, backpressure, stats
├── logs/                         # Rotating log output (mounted volume in Docker)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```
