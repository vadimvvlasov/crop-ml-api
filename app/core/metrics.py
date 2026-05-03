from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status_code"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "path"],
    registry=registry,
)

inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Model inference duration in seconds",
    registry=registry,
)

__all__ = [
    "registry",
    "generate_latest",
    "CONTENT_TYPE_LATEST",
    "http_requests_total",
    "http_request_duration_seconds",
    "inference_duration_seconds",
]
