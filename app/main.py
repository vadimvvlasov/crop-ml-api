import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .core.logging import configure_logging
from .model import CLASS_NAMES, load_model, predict
from .sample import make_sample
from .schemas import Prediction, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.model = load_model()
    logger.info(
        "model_loaded",
        extra={"event": "model_loaded"},
    )
    yield


app = FastAPI(title="Crop Prediction API", version="0.1.0", lifespan=lifespan)


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
def predict_endpoint(req: PredictRequest, request: Request) -> PredictResponse:
    class_ids, probas = predict(
        req.features, req.week_of_year, req.location, model=request.app.state.model
    )
    return _to_response(class_ids, probas)


@app.post("/predict/demo", response_model=PredictResponse)
def predict_demo(request: Request, n: int = 2) -> PredictResponse:
    data = make_sample(n)
    class_ids, probas = predict(
        data["features"],
        data["week_of_year"],
        data["location"],
        model=request.app.state.model,
    )
    return _to_response(class_ids, probas)
