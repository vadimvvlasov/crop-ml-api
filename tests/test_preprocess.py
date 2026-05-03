"""Unit tests for HLS NaN interpolation (no model load required)."""

import numpy as np

from app.hls_preprocess import interpolate_features_ntc

N, T, C = 3, 26, 15


def _ones() -> np.ndarray:
    return np.ones((N, T, C), dtype=np.float32)


def test_shape_preserved():
    out = interpolate_features_ntc(_ones())
    assert out.shape == (N, T, C)


def test_no_nan_clean_input():
    out = interpolate_features_ntc(_ones())
    assert not np.isnan(out).any()


def test_no_nan_all_nan_input():
    arr = np.full((N, T, C), np.nan, dtype=np.float32)
    out = interpolate_features_ntc(arr)
    assert not np.isnan(out).any()


def test_no_nan_partial_gap_middle():
    arr = _ones()
    arr[0, 5:10, :] = np.nan  # gap in the middle — interpolatable
    out = interpolate_features_ntc(arr)
    assert not np.isnan(out).any()


def test_no_nan_gap_at_start():
    arr = _ones()
    arr[:, :3, :] = np.nan  # leading NaNs — needs bfill
    out = interpolate_features_ntc(arr)
    assert not np.isnan(out).any()


def test_no_nan_gap_at_end():
    arr = _ones()
    arr[:, -3:, :] = np.nan  # trailing NaNs — needs ffill
    out = interpolate_features_ntc(arr)
    assert not np.isnan(out).any()


def test_values_unchanged_where_not_nan():
    arr = _ones() * 5.0
    arr[0, 10, 0] = np.nan
    out = interpolate_features_ntc(arr)
    # Non-NaN values should stay at 5.0 (linear interp between equal values)
    assert np.allclose(out[0, 9, 0], 5.0)
    assert np.allclose(out[0, 11, 0], 5.0)


def test_output_dtype_is_float32():
    out = interpolate_features_ntc(_ones())
    assert out.dtype == np.float32


def test_single_row():
    arr = np.random.rand(1, T, C).astype(np.float32)
    arr[0, 0, :] = np.nan
    out = interpolate_features_ntc(arr)
    assert out.shape == (1, T, C)
    assert not np.isnan(out).any()
