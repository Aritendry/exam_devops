import pytest
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.train import DEFAULT_DATA_PATH, EXAMPLE_RECORD, train_model


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_and_prediction_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("AUTO_TRAIN_ON_STARTUP", "false")

    train_model(
        data_path=DEFAULT_DATA_PATH,
        artifact_dir=tmp_path,
        n_estimators=10,
        max_depth=6,
        enable_mlflow=False,
    )

    app = create_app()

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health_response = await client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["model_loaded"] is True

            predict_response = await client.post("/predict", json=EXAMPLE_RECORD)
            assert predict_response.status_code == 200
            payload = predict_response.json()
            assert payload["model_name"] == "student-score-predictor"
            assert isinstance(payload["prediction"], float)
