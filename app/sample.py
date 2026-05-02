"""Demo payloads built from real RS fixtures (interpolated HLS batches)."""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "rs_hls_predict_request.json"
RNG = random.Random(42)


@lru_cache(maxsize=1)
def _fixture_payload() -> dict:
    if not _FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"Fixture missing: {_FIXTURE_PATH}. Run scripts/export_rs_hls_fixtures.py."
        )
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _pick_indices(n: int, pool: int) -> list[int]:
    if pool < 1:
        raise ValueError("Fixture batch is empty")
    if n <= pool:
        return RNG.sample(range(pool), n)
    return RNG.choices(range(pool), k=n)


def make_sample(n: int = 2) -> dict:
    """Random rows from ``fixtures/rs_hls_predict_request.json`` (same ``week_of_year`` for all)."""
    if n < 1:
        raise ValueError("n must be >= 1")

    data = _fixture_payload()
    feats_all = data["features"]
    loc_all = data["location"]
    pool = len(feats_all)
    ix = _pick_indices(n, pool)

    return {
        "features": [feats_all[i] for i in ix],
        "week_of_year": list(data["week_of_year"]),
        "location": [loc_all[i] for i in ix],
    }
