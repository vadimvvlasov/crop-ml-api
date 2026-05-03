"""ASGI observability middleware for crop-ml-api.

Intercepts every HTTP request to:
- Generate a UUID v4 ``request_id`` and attach it as the ``X-Request-ID``
  response header.
- Log a structured JSON record (INFO) with request metadata after the
  response is produced.
- Increment Prometheus counters / histograms for all paths except those
  listed in ``EXCLUDED_PATHS``.
- Log an ERROR record (with exception details) for any unhandled exception
  before re-raising it so FastAPI can return a 500 response.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

logger = logging.getLogger(__name__)

# Paths excluded from Prometheus metric collection.
# Logging is still performed for these paths.
EXCLUDED_PATHS: frozenset[str] = frozenset({"/metrics"})


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request-id tracking, structured logging, and
    Prometheus metrics to every HTTP request handled by the application."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            logger.error(
                "Unhandled exception",
                extra={
                    "request_id": request_id,
                    "exc_type": type(exc).__name__,
                    "exc_message": str(exc),
                },
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        if request.url.path not in EXCLUDED_PATHS:
            http_requests_total.labels(
                method=request.method,
                path=request.url.path,
                status_code=str(status_code),
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                path=request.url.path,
            ).observe(duration_ms / 1000)

        response.headers["X-Request-ID"] = request_id
        return response
