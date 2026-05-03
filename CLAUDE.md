# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install deps (CPU-only torch via pytorch-cpu index)
uv sync

# Run dev server
uv run uvicorn app.main:app --reload --port 8000

# Run with dev dependencies (for fixture export)
uv sync --group dev

# Export RS fixtures from research-crops
RESEARCH_CROPS_ROOT=/path/to/research-crops uv run python scripts/export_rs_hls_fixtures.py

# Smoke test
curl -sf -X POST -H "Content-Type: application/json" \
  -d @fixtures/rs_hls_predict_request.json http://localhost:8000/predict | jq

```

No test suite yet. M1 will add pytest.

## Architecture

MVP FastAPI inference service for HLS Transformer crop classifier (Soybean/Corn/Rice/Other).

### Request Flow

```
POST /predict
  → PredictRequest (Pydantic validates shape N,26,15)
  → model.predict()
    → interpolate_features_ntc() fills NaN (matches research-crops HLSStitchedDataset)
    → normalize by 0.7e-4, convert to tensor
    → TransformerModel.forward(x, week_of_year, location)
  → class_id + class_name + proba
```

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, 4 routes: `/health`, `/sample`, `/predict`, `/predict/demo` |
| `app/model.py` | `lru_cache` model loader + `predict()` — M1 replaces with lifespan |
| `app/schemas.py` | Pydantic `PredictRequest`/`PredictResponse` with shape validation via `model_validator` |
| `app/hls_preprocess.py` | NaN interpolation matching `research-crops` `HLSStitchedDataset` |
| `app/sample.py` | Random rows from `fixtures/rs_hls_predict_request.json` |
| `src/models/transformer_model.py` | TL-Transformer architecture — mirror required for `torch.load(weights_only=False)` pickle resolution |
| `scripts/export_rs_hls_fixtures.py` | Export fixtures from `research-crops` (needs `RESEARCH_CROPS_ROOT` env var) |

### TL-Transformer Input/Output

- **Input**: `(N, T=26, C=15)` reflectance + `(T,) week_of_year` + `(N,2) location`
- **Architecture**: SpectralProjection → TemporalEncoding + LocationEncoding → TransformerEncoder (5 layers, 2 heads, d_model=182) → MaxPool → Linear → LogSoftmax
- **Output**: `(N, 4)` log-probabilities

### Critical Design Decisions

- `torch.load(weights_only=False)` requires `src.models.transformer_model` as mirror — class path `src.models.transformer_model.TransformerModel` embedded in pickle
- Shape validation in Pydantic `model_validator` prevents `mat1` shape errors
- Interpolation order must match `research-crops`: interpolate first, then `0.7e-4` normalization
- `fixtures/rs_hls_predict_request.json` is API contract — regenerate with script if model input changes
- CPU-only torch via `pyproject.toml` `[tool.uv.sources]` + pytorch-cpu index (skips ~2GB CUDA wheels)

## Milestones (M0-M4)

Skills/prompts in `prompts/`. Each milestone has plan-mode paste in README.md.

| Milestone | Focus |
|-----------|-------|
| M0 (current) | Baseline: FastAPI + uv + predict + fixtures |
| M1 | Docker, lifespan, structlog, Prometheus `/metrics` |
| M2 | Postgres async, SQLAlchemy 2.0, Alembic, Redis idempotency |
| M3 | AsyncBatcher, OTel traces, timeouts, graceful shutdown |
| M4 | LLM explainer, prompt versioning, circuit breaker, CI/CD |

Branch naming: `feature/m1-observability`, `feature/m2-persistence`, etc.

## Verification

```bash
# Health check
curl -s http://localhost:8000/health

# Demo predict
curl -s -X POST http://localhost:8000/predict/demo | jq

# Full fixture smoke test (expect 10 predictions, finite probas)
curl -s http://localhost:8000/sample > /tmp/payload.json
curl -sf -X POST -H "Content-Type: application/json" \
  -d @fixtures/rs_hls_predict_request.json http://localhost:8000/predict | jq '.predictions | length'
```

OpenAPI UI: http://localhost:8000/docs

## M0 Exit Checklist

- `/health` returns 200
- `/predict` with fixture returns 10 predictions
- No `NaN` in fixture JSON (`export` uses `allow_nan=False`)

## RS Fixture Details

Regenerated from `research-crops` (Rio Grande do Sul bundle). `crop_class` from GPKG mapped via `classmapping_eval.csv`; `Pasture` / `Forest Plantation` → `Other` (class_id=2).

Quick win for M1: replace `lru_cache` with FastAPI `lifespan` so model warms at startup (load errors surface on boot, not first request).
