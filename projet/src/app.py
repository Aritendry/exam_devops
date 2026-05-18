from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_records
from src.train import resolve_artifact_dir

REQUEST_COUNT = Counter(
    "student_api_requests_total",
    "Nombre total de requetes HTTP exposees par l'API.",
)
PREDICTION_COUNT = Counter(
    "student_api_predictions_total",
    "Nombre total de predictions effectuees.",
)
PREDICTION_LATENCY = Histogram(
    "student_api_prediction_seconds",
    "Temps de reponse des predictions.",
)


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Hours_Studied: float = Field(..., ge=0)
    Attendance: float = Field(..., ge=0, le=100)
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Previous_Scores: float = Field(..., ge=0, le=100)
    Tutoring_Sessions: int = Field(..., ge=0)
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Motivation_Level: str
    Internet_Access: str

    def to_record(self) -> dict[str, object]:
        return self.model_dump()


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[StudentFeatures] = Field(..., min_length=1)


def create_app() -> FastAPI:
    artifact_dir = resolve_artifact_dir()
    auto_train = os.getenv("AUTO_TRAIN_ON_STARTUP", "true").lower() == "true"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifact_dir = Path(artifact_dir)
        app.state.startup_error = None
        try:
            model, metadata = load_model(
                artifact_dir=artifact_dir,
                auto_train=auto_train,
            )
            app.state.model = model
            app.state.metadata = metadata
        except Exception as exc:  # pragma: no cover
            app.state.model = None
            app.state.metadata = None
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title="Student Score MLOps API",
        version="1.0.0",
        description="API locale de prediction des notes finales d'etudiants.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    async def read_index():
        REQUEST_COUNT.inc()
        with open("src/index.html", "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/health")
    async def health() -> dict[str, object]:
        REQUEST_COUNT.inc()
        return {
            "status": "ok" if app.state.model is not None else "degraded",
            "model_loaded": app.state.model is not None,
            "artifact_dir": str(app.state.artifact_dir),
            "startup_error": app.state.startup_error,
        }

    @app.get("/model/info")
    async def model_info() -> dict[str, object]:
        REQUEST_COUNT.inc()
        if app.state.metadata is None:
            raise HTTPException(status_code=503, detail="Le modele n'est pas charge.")
        return app.state.metadata

    @app.post("/predict")
    async def predict(payload: StudentFeatures) -> dict[str, object]:
        REQUEST_COUNT.inc()
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Le modele n'est pas disponible.",
            )
        start_time = perf_counter()
        result = predict_records(
            records=[payload.to_record()],
            model=app.state.model,
            metadata=app.state.metadata,
        )
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(perf_counter() - start_time)
        return {
            "prediction": result["predictions"][0],
            "model_name": result["model_name"],
        }

    @app.post("/predict/batch")
    async def predict_batch(payload: BatchPredictionRequest) -> dict[str, object]:
        REQUEST_COUNT.inc()
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Le modele n'est pas disponible.",
            )
        start_time = perf_counter()
        result = predict_records(
            records=[record.to_record() for record in payload.records],
            model=app.state.model,
            metadata=app.state.metadata,
        )
        PREDICTION_COUNT.inc(len(payload.records))
        PREDICTION_LATENCY.observe(perf_counter() - start_time)
        return result

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
