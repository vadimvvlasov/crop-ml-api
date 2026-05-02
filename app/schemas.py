from typing import List, Optional

from pydantic import BaseModel, Field


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


class Prediction(BaseModel):
    class_id: int
    class_name: Optional[str]
    proba: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
