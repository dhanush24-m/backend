"""
Structured JSON logging with:
  - Console output
  - Rotating file output (configurable size / backup count)
  - Session / request IDs injected into every record
"""

import json
import logging
import logging.handlers
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from app.config import get_settings

# ── Context variable so per-request metadata travels through async calls ──────
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def set_logging_context(session_id: str = "", request_id: str = "") -> None:
    _session_id_var.set(session_id)
    _request_id_var.set(request_id)


def new_request_id() -> str:
    rid = str(uuid.uuid4())[:8]
    _request_id_var.set(rid)
    return rid


# ── JSON formatter ─────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "session_id": _session_id_var.get() or record.__dict__.get("session_id", ""),
            "request_id": _request_id_var.get() or record.__dict__.get("request_id", ""),
            "msg": record.getMessage(),
        }
        # Carry any extra keys set via `logger.info("...", extra={...})`
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            } and not key.startswith("_"):
                payload[key] = val

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove default handlers
    root.handlers.clear()

    formatter = JsonFormatter()

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Rotating file
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=settings.log_rotation_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
