from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

# HLS Transformer contract (must match training / fixtures)
T_TIMESTEPS = 26
N_CHANNELS = 15


class PredictRequest(BaseModel):
    features: List[List[List[float]]] = Field(
        ..., description="Raw HLS reflectance, shape (N, T=26, C=15)"
    )
    week_of_year: List[int] = Field(
        ..., description="ISO week per timestep, shape (T=26,)"
    )
    location: List[List[float]] = Field(
        ..., description="Field centroids, shape (N, 2) as [lat, lon]"
    )

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "PredictRequest":
        n = len(self.features)
        if n < 1:
            raise ValueError("features: batch size N must be >= 1")
        if len(self.location) != n:
            raise ValueError(
                f"location length ({len(self.location)}) must match features batch N ({n})"
            )
        if len(self.week_of_year) != T_TIMESTEPS:
            raise ValueError(
                f"week_of_year must have length {T_TIMESTEPS}, got {len(self.week_of_year)}"
            )
        for i, time_rows in enumerate(self.features):
            if len(time_rows) != T_TIMESTEPS:
                raise ValueError(
                    f"features[{i}] must have {T_TIMESTEPS} timesteps, got {len(time_rows)}"
                )
            for t, band_row in enumerate(time_rows):
                if len(band_row) != N_CHANNELS:
                    raise ValueError(
                        f"features[{i}][{t}] must have {N_CHANNELS} bands, "
                        f"got {len(band_row)}"
                    )
        for i, ll in enumerate(self.location):
            if len(ll) != 2:
                raise ValueError(f"location[{i}] must be [lat, lon] (length 2)")
        return self


class Prediction(BaseModel):
    class_id: int
    class_name: Optional[str]
    proba: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
