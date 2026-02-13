"""
Logging middleware and utilities.

Provides:
- Request ID generation and propagation
- Structured JSON logging
- Log context management
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Context variable for request ID
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# Module logger
logger = logging.getLogger("model_lab")


def get_request_id() -> str | None:
    """Get current request ID from context."""
    return request_id_ctx.get()


def set_request_id(req_id: str) -> None:
    """Set request ID in context."""
    request_id_ctx.set(req_id)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a unique request ID to each request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Set in context
        request_id_ctx.set(request_id)

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def log_request(
    method: str,
    path: str,
    status_code: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an API request with structured data."""
    req_id = get_request_id() or "none"

    log_data = {
        "request_id": req_id,
        "method": method,
        "path": path,
    }

    if status_code is not None:
        log_data["status_code"] = status_code

    if error:
        log_data["error"] = error

    if extra:
        log_data.update(extra)

    if status_code and status_code >= 500:
        logger.error(f"{method} {path} failed: {error}", extra=log_data)
    elif status_code and status_code >= 400:
        logger.warning(f"{method} {path} client error: {error}", extra=log_data)
    else:
        logger.info(f"{method} {path} {status_code or 'pending'}", extra=log_data)


def log_run_event(run_id: str, event: str, extra: dict[str, Any] | None = None) -> None:
    """Log a run-related event."""
    req_id = get_request_id() or "none"

    log_data = {
        "request_id": req_id,
        "run_id": run_id,
        "event": event,
    }

    if extra:
        log_data.update(extra)

    logger.info(f"Run {run_id}: {event}", extra=log_data)
