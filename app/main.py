from fastapi import FastAPI

from .model import CLASS_NAMES, predict
from .sample import make_sample
from .schemas import Prediction, PredictRequest, PredictResponse

app = FastAPI(title="Crop Prediction API", version="0.1.0")


def _to_response(class_ids: list[int], probas: list[float]) -> PredictResponse:
    return PredictResponse(
        predictions=[
            Prediction(class_id=c, class_name=CLASS_NAMES.get(c), proba=p)
            for c, p in zip(class_ids, probas)
        ]
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sample", response_model=PredictRequest)
def sample(n: int = 2) -> dict:
    return make_sample(n)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    class_ids, probas = predict(req.features, req.week_of_year, req.location)
    return _to_response(class_ids, probas)


@app.post("/predict/demo", response_model=PredictResponse)
def predict_demo(n: int = 2) -> PredictResponse:
    data = make_sample(n)
    class_ids, probas = predict(data["features"], data["week_of_year"], data["location"])
    return _to_response(class_ids, probas)
