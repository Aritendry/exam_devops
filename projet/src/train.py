from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "student_dataset.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "student_score_pipeline.joblib"
DEFAULT_METADATA_PATH = DEFAULT_ARTIFACT_DIR / "model_metadata.json"
DEFAULT_MLFLOW_DIR = PROJECT_ROOT / "mlruns"

TARGET_COLUMN = "Final_Exam_Score"
NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
]
CATEGORICAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXAMPLE_RECORD = {
    "Hours_Studied": 22.0,
    "Attendance": 91.0,
    "Sleep_Hours": 7.4,
    "Previous_Scores": 74.0,
    "Tutoring_Sessions": 2,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
}


def resolve_data_path(candidate: str | Path | None = None) -> Path:
    env_value = os.getenv("DATASET_PATH")
    path = Path(candidate or env_value or DEFAULT_DATA_PATH)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_artifact_dir(candidate: str | Path | None = None) -> Path:
    env_value = os.getenv("MODEL_ARTIFACT_DIR")
    path = Path(candidate or env_value or DEFAULT_ARTIFACT_DIR)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_encoder() -> OneHotEncoder:
    encoder_kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        encoder_kwargs["sparse_output"] = False
    else:  # pragma: no cover
        encoder_kwargs["sparse"] = False
    return OneHotEncoder(**encoder_kwargs)


def build_pipeline(
    n_estimators: int = 80,
    max_depth: int = 12,
    random_state: int = 42,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    resolved_path = resolve_data_path(data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {resolved_path}")
    return pd.read_csv(resolved_path)


def _prepare_tracking_uri() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri
    DEFAULT_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_MLFLOW_DIR.resolve().as_uri()


def train_model(
    data_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    n_estimators: int = 80,
    max_depth: int = 12,
    random_state: int = 42,
    test_size: float = 0.2,
    enable_mlflow: bool | None = None,
) -> dict[str, Any]:
    resolved_data_path = resolve_data_path(data_path)
    resolved_artifact_dir = resolve_artifact_dir(artifact_dir)
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(resolved_data_path)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    tracking_enabled = enable_mlflow
    if tracking_enabled is None:
        tracking_enabled = os.getenv("ENABLE_MLFLOW", "false").lower() == "true"
    tracking_enabled = bool(tracking_enabled and mlflow is not None)

    run_started = False
    tracking_uri = _prepare_tracking_uri()
    if tracking_enabled:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("student-score-local")
        mlflow.start_run(
            run_name=f"rf-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        run_started = True

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metadata = {
            "model_name": "student-score-predictor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(resolved_data_path),
            "artifact_dir": str(resolved_artifact_dir),
            "target_column": TARGET_COLUMN,
            "feature_columns": FEATURE_COLUMNS,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "test_size": test_size,
            },
            "metrics": {
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
                "r2": round(float(r2), 4),
            },
            "mlflow": {
                "enabled": tracking_enabled,
                "tracking_uri": tracking_uri if tracking_enabled else None,
            },
            "example_record": EXAMPLE_RECORD,
        }

        model_path = resolved_artifact_dir / DEFAULT_MODEL_PATH.name
        metadata_path = resolved_artifact_dir / DEFAULT_METADATA_PATH.name
        joblib.dump(pipeline, model_path)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        if tracking_enabled:
            mlflow.log_params(metadata["hyperparameters"])
            mlflow.log_metrics(metadata["metrics"])
            mlflow.log_dict(metadata, "model_metadata.json")
            mlflow.sklearn.log_model(pipeline, "student_score_model")
            metadata["mlflow"]["run_id"] = mlflow.active_run().info.run_id
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

        return metadata
    finally:
        if run_started:
            mlflow.end_run()


def ensure_model_artifacts(
    artifact_dir: str | Path | None = None,
    data_path: str | Path | None = None,
    enable_mlflow: bool | None = None,
) -> dict[str, Any]:
    resolved_artifact_dir = resolve_artifact_dir(artifact_dir)
    model_path = resolved_artifact_dir / DEFAULT_MODEL_PATH.name
    metadata_path = resolved_artifact_dir / DEFAULT_METADATA_PATH.name

    if model_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    return train_model(
        data_path=data_path,
        artifact_dir=resolved_artifact_dir,
        enable_mlflow=enable_mlflow,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrainement local du modele de prediction de score final."
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Chemin du dataset CSV.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Dossier de sortie des artefacts.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(os.getenv("MODEL_N_ESTIMATORS", "80")),
        help="Nombre d'arbres pour la foret aleatoire.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=int(os.getenv("MODEL_MAX_DEPTH", "12")),
        help="Profondeur maximale des arbres.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("MODEL_RANDOM_STATE", "42")),
        help="Graine pseudo-aleatoire.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Desactive le tracking MLflow meme si disponible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train_model(
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        enable_mlflow=not args.disable_mlflow,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
