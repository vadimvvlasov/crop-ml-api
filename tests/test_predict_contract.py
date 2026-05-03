"""Regression / contract tests for POST /predict."""

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import N_CHANNELS, T_TIMESTEPS

client = TestClient(app)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rs_hls_predict_request.json"
)


@pytest.fixture(scope="module")
def fixture_payload() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Happy path — fixture payload (N=10)
# ---------------------------------------------------------------------------


def test_fixture_predict_count(fixture_payload):
    """POST /predict with the fixture payload (N=10) returns exactly 10 predictions."""
    r = client.post("/predict", json=fixture_payload)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 10


def test_fixture_predict_proba_finite(fixture_payload):
    """All proba values from the fixture response are finite."""
    r = client.post("/predict", json=fixture_payload)
    assert r.status_code == 200
    for p in r.json()["predictions"]:
        assert math.isfinite(p["proba"]), f"proba is not finite: {p['proba']}"


# ---------------------------------------------------------------------------
# Validation errors (422) — wrong feature shapes
# ---------------------------------------------------------------------------


def test_predict_422_wrong_timesteps(fixture_payload):
    """Features with wrong number of timesteps (10 instead of 26) → HTTP 422."""
    wrong_t = 10
    n = len(fixture_payload["features"])
    bad_features = [[[0.0] * N_CHANNELS for _ in range(wrong_t)] for _ in range(n)]
    bad_payload = {
        "features": bad_features,
        "week_of_year": fixture_payload["week_of_year"],  # still 26 items
        "location": fixture_payload["location"],
    }
    assert client.post("/predict", json=bad_payload).status_code == 422


def test_predict_422_wrong_channels(fixture_payload):
    """Features with wrong number of channels (5 instead of 15) → HTTP 422."""
    wrong_c = 5
    n = len(fixture_payload["features"])
    bad_features = [[[0.0] * wrong_c for _ in range(T_TIMESTEPS)] for _ in range(n)]
    bad_payload = {
        "features": bad_features,
        "week_of_year": fixture_payload["week_of_year"],
        "location": fixture_payload["location"],
    }
    assert client.post("/predict", json=bad_payload).status_code == 422
