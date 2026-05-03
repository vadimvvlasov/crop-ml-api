# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies (cached separately from code)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && chmod -R a+rX /app/.venv

# Source code
COPY app/ ./app/
COPY src/ ./src/
COPY fixtures/ ./fixtures/

# Model weights (included in image for production)
COPY models/ ./models/

# Non-root user
RUN adduser --system --no-create-home appuser
USER appuser

EXPOSE 8000

ENV UV_NO_CACHE=1

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
