from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from .hls_preprocess import interpolate_features_ntc

# Mirror of the pickle-time module path `src.models.transformer_model`.
# Importing it ensures `torch.load` (full pickle, not state_dict) can resolve
# the `TransformerModel` class without sys.path tricks.
import src.models.transformer_model  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hls_TL_dinner1024_nhead2_nlayers5_260311_1423.pt"

# Source of truth: research-crops/dags/data/hls/external/classmapping_eval.csv
CLASS_NAMES = {0: "Soybean", 1: "Corn", 2: "Other", 3: "Rice"}

# Must match training: scales raw HLS reflectance into model's input range.
NORMALIZATION_FACTOR = 0.7e-4


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    return model


def predict(
    features: Sequence,
    week_of_year: Sequence[int],
    location: Sequence[Sequence[float]],
) -> Tuple[list[int], list[float]]:
    """Run inference on raw HLS features. Returns (class_ids, probas) of length N."""
    model = get_model()
    device = next(model.parameters()).device

    filled = interpolate_features_ntc(np.asarray(features, dtype=np.float32))
    x = torch.as_tensor(filled, dtype=torch.float32, device=device) * NORMALIZATION_FACTOR
    week = torch.as_tensor(week_of_year, dtype=torch.float32, device=device)
    loc = torch.as_tensor(location, dtype=torch.float32, device=device)

    with torch.no_grad():
        logp = model(x, week_of_year=week, location=loc)
        top = logp.exp().max(dim=-1)
    return top.indices.cpu().tolist(), top.values.cpu().tolist()
