"""Integration tests for FastAPI endpoints (loads real model weights)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rs_hls_predict_request.json"
)

VALID_CLASS_NAMES = {"Soybean", "Corn", "Rice", "Other"}


@pytest.fixture(scope="module")
def fixture_payload() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /predict  — happy path
# ---------------------------------------------------------------------------


def test_predict_fixture_status(fixture_payload):
    r = client.post("/predict", json=fixture_payload)
    assert r.status_code == 200


def test_predict_fixture_count(fixture_payload):
    r = client.post("/predict", json=fixture_payload)
    preds = r.json()["predictions"]
    assert len(preds) == len(fixture_payload["features"])


def test_predict_fixture_proba_range(fixture_payload):
    r = client.post("/predict", json=fixture_payload)
    for p in r.json()["predictions"]:
        assert 0.0 <= p["proba"] <= 1.0, f"proba out of range: {p['proba']}"


def test_predict_fixture_class_names(fixture_payload):
    r = client.post("/predict", json=fixture_payload)
    for p in r.json()["predictions"]:
        assert p["class_name"] in VALID_CLASS_NAMES


def test_predict_fixture_no_nan_proba(fixture_payload):
    r = client.post("/predict", json=fixture_payload)
    for p in r.json()["predictions"]:
        assert p["proba"] == p["proba"], "proba is NaN"  # NaN != NaN


# ---------------------------------------------------------------------------
# /predict  — validation errors (422)
# ---------------------------------------------------------------------------


def test_predict_wrong_week_length(fixture_payload):
    bad = {**fixture_payload, "week_of_year": fixture_payload["week_of_year"][:10]}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_empty_features(fixture_payload):
    bad = {**fixture_payload, "features": [], "location": []}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_location_mismatch(fixture_payload):
    bad = {**fixture_payload, "location": fixture_payload["location"][:1]}
    assert client.post("/predict", json=bad).status_code == 422


# ---------------------------------------------------------------------------
# /predict/demo
# ---------------------------------------------------------------------------


def test_predict_demo_default():
    r = client.post("/predict/demo")
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_demo_n_param():
    r = client.post("/predict/demo?n=4")
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 4


# ---------------------------------------------------------------------------
# /sample
# ---------------------------------------------------------------------------


def test_sample_default_shape():
    r = client.get("/sample")
    assert r.status_code == 200
    data = r.json()
    assert len(data["features"]) == 2
    assert len(data["week_of_year"]) == 26
    assert len(data["location"]) == 2


def test_sample_n_param():
    r = client.get("/sample?n=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data["features"]) == 5
    assert len(data["location"]) == 5
