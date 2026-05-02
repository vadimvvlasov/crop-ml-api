# crop-ml-api

MVP FastAPI inference service for the HLS Transformer crop classifier (Soybean / Corn / Rice / Other).

This iteration covers Week 1 (MVP) only — see [prompts/crop-ml-api_spec.md](prompts/crop-ml-api_spec.md) for the full roadmap (Postgres, Redis, AsyncBatcher, OTel, LLM explainer, Caddy/CI deploy).

## Layout

```
app/
  main.py        # FastAPI app + 4 routes
  model.py       # lru_cache loader + predict()
  schemas.py     # Pydantic Request/Response
  sample.py      # random rows from fixtures/rs_hls_predict_request.json
src/models/      # pickle-path mirror so torch.load resolves TransformerModel
models/          # weights (.pt)
prompts/         # source spec + minimal-project spec
```

The `src/models/transformer_model.py` mirror is required because the model
is saved as a full pickle (`torch.load(weights_only=False)`) and embeds the
fully-qualified class path `src.models.transformer_model.TransformerModel`.

## Run

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

## Verify

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict/demo | jq
curl -s http://localhost:8000/sample > payload.json
curl -s -X POST -H "Content-Type: application/json" -d @payload.json http://localhost:8000/predict | jq
```

OpenAPI UI: <http://localhost:8000/docs>.

## Real RS fixtures (NPZ + GPKG)

Regenerate `fixtures/rs_hls_predict_request.json` + `fixtures/rs_hls_fixture_meta.json`
from `research-crops` (Rio Grande do Sul bundle). ``crop_class`` from GPKG is mapped to
eval IDs via ``classmapping_eval.csv``; ``Pasture`` / ``Forest Plantation`` → ``Other`` (2).

```bash
uv sync --group dev
RESEARCH_CROPS_ROOT=/path/to/research-crops uv run python scripts/export_rs_hls_fixtures.py
curl -s -X POST -H "Content-Type: application/json" \
  -d @fixtures/rs_hls_predict_request.json http://localhost:8000/predict | jq
```

## Inputs

`POST /predict`:

- `features`: `(N, T=26, C=15)` raw HLS reflectance (~`[0..10000]`). Missing values (`NaN`) are linearly interpolated along time then ffill/bfill, matching `research-crops` `HLSStitchedDataset`; then multiplied by `0.7e-4`.
- `week_of_year`: `(T=26,)` ISO week per timestep.
- `location`: `(N, 2)` — `[lat, lon]` field centroids.

Returns log-softmax → `class_id` + `class_name` + `proba` per row.

## Roadmap

Future iterations from [the spec](prompts/crop-ml-api_spec.md):
Postgres + SQLAlchemy + Alembic, Redis idempotency, `AsyncBatcher`,
OpenTelemetry, structlog, Prometheus `/metrics`, LLM explainer with
prompt versioning + cost tracking + fallback, Caddy + Docker compose,
GitHub Actions CI/CD, load tests.

Quick win for next step: replace `lru_cache` with FastAPI `lifespan` so
the model is warmed up at startup (load errors surface on boot, not on
first request).
