"""Tests for health endpoint, X-Request-ID header, and /metrics endpoint.

Covers:
- Requirements 2.2: X-Request-ID header on every response
- Requirements 3.1: /metrics endpoint in Prometheus text format
"""

import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok():
    """GET /health returns {"status": "ok"} with HTTP 200."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_has_request_id_header():
    """Response from GET /health contains the X-Request-ID header."""
    r = client.get("/health")
    assert "x-request-id" in r.headers


def test_request_id_is_uuid4():
    """X-Request-ID value is a valid UUID v4."""
    r = client.get("/health")
    value = r.headers.get("x-request-id")
    assert value is not None, "X-Request-ID header is missing"
    parsed = uuid.UUID(value, version=4)
    # uuid.UUID normalises the version bits; confirm the string round-trips
    assert str(parsed) == value


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200():
    """GET /metrics returns HTTP 200."""
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_content_type():
    """Content-Type of /metrics response contains 'text/plain'."""
    r = client.get("/metrics")
    assert "text/plain" in r.headers.get("content-type", "")
