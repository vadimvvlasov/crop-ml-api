"""HLS gap filling aligned with ``research-crops/src/data/hls_stitch.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_timeseries_all_tnc(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation along time (axis 0) for ``(T, N, C)``.

    Same logic as ``HLSStitchedDataset`` / ``_interpolate_timeseries_all``.
    """
    t, n, c = arr.shape
    result = np.empty((t, n, c), dtype=np.float32)
    for band in range(c):
        df = pd.DataFrame(arr[:, :, band])
        df = df.interpolate(method="linear", axis=0)
        df = df.ffill(axis=0).bfill(axis=0)
        result[:, :, band] = df.fillna(0.0).to_numpy(dtype=np.float32)
    return result


def interpolate_features_ntc(features_ntc: np.ndarray) -> np.ndarray:
    """Fill NaNs in raw reflectance ``(N, T, C)``. Does not apply reflectance scale."""
    arr = np.asarray(features_ntc, dtype=np.float32)
    tnc = np.transpose(arr, (1, 0, 2))
    filled_tnc = interpolate_timeseries_all_tnc(tnc)
    return np.transpose(filled_tnc, (1, 0, 2)).astype(np.float32, copy=False)
