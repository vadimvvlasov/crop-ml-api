"""Unit tests for Pydantic shape validation (no model load required)."""

import pytest
from pydantic import ValidationError

from app.schemas import N_CHANNELS, T_TIMESTEPS, PredictRequest


def _valid_payload(n: int = 2) -> dict:
    return {
        "features": [[[0.0] * N_CHANNELS] * T_TIMESTEPS] * n,
        "week_of_year": list(range(1, T_TIMESTEPS + 1)),
        "location": [[-15.0, -50.0]] * n,
    }


def test_valid_single_row():
    PredictRequest(**_valid_payload(n=1))


def test_valid_batch():
    PredictRequest(**_valid_payload(n=5))


def test_wrong_timesteps_raises():
    p = _valid_payload()
    p["features"][0] = p["features"][0][: T_TIMESTEPS - 1]  # 25 instead of 26
    with pytest.raises(ValidationError):
        PredictRequest(**p)


def test_wrong_channels_raises():
    p = _valid_payload()
    p["features"][0][0] = p["features"][0][0][: N_CHANNELS - 1]  # 14 instead of 15
    with pytest.raises(ValidationError):
        PredictRequest(**p)


def test_location_mismatch_raises():
    p = _valid_payload(n=2)
    p["location"] = p["location"][:1]  # n=2 features but only 1 location
    with pytest.raises(ValidationError):
        PredictRequest(**p)


def test_week_wrong_length_raises():
    p = _valid_payload()
    p["week_of_year"] = p["week_of_year"][:10]
    with pytest.raises(ValidationError):
        PredictRequest(**p)


def test_empty_batch_raises():
    p = _valid_payload()
    p["features"] = []
    p["location"] = []
    with pytest.raises(ValidationError):
        PredictRequest(**p)


def test_location_wrong_length_raises():
    p = _valid_payload(n=1)
    p["location"] = [[-15.0]]  # missing lon
    with pytest.raises(ValidationError):
        PredictRequest(**p)
