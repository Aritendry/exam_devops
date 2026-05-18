from src.train import DEFAULT_DATA_PATH, train_model


def test_train_model_creates_artifacts(tmp_path):
    metadata = train_model(
        data_path=DEFAULT_DATA_PATH,
        artifact_dir=tmp_path,
        n_estimators=10,
        max_depth=6,
        enable_mlflow=False,
    )

    assert metadata["model_name"] == "student-score-predictor"
    assert metadata["metrics"]["rmse"] > 0
    assert (tmp_path / "student_score_pipeline.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()
