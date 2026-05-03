import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .api.middleware import ObservabilityMiddleware
from .core.logging import configure_logging
from .core.metrics import registry
from .model import CLASS_NAMES, MODEL_PATH, load_model, predict
from .sample import make_sample
from .schemas import Prediction, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.model = load_model()  # FileNotFoundError → процесс падает
    logger.info(
        "model_loaded",
        extra={"event": "model_loaded", "model_path": str(MODEL_PATH)},
    )
    yield


app = FastAPI(title="Crop Prediction API", version="0.1.0", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)


def _to_response(class_ids: list[int], probas: list[float]) -> PredictResponse:
    return PredictResponse(
        predictions=[
            Prediction(class_id=c, class_name=CLASS_NAMES.get(c), proba=p)
            for c, p in zip(class_ids, probas)
        ]
    )


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


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
