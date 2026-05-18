from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.train import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    EXAMPLE_RECORD,
    FEATURE_COLUMNS,
    ensure_model_artifacts,
    resolve_artifact_dir,
)


def load_model(
    artifact_dir: str | Path | None = None,
    auto_train: bool = True,
) -> tuple[Any, dict[str, Any]]:
    resolved_artifact_dir = resolve_artifact_dir(artifact_dir)
    model_path = resolved_artifact_dir / DEFAULT_MODEL_PATH.name
    metadata_path = resolved_artifact_dir / DEFAULT_METADATA_PATH.name

    if auto_train:
        ensure_model_artifacts(artifact_dir=resolved_artifact_dir, enable_mlflow=False)

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Artefacts du modele absents. Lancez `python -m src.train` d'abord."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour la prediction: " + ", ".join(missing_columns)
        )
    return frame[FEATURE_COLUMNS]


def predict_records(
    records: list[dict[str, Any]],
    artifact_dir: str | Path | None = None,
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model is None or metadata is None:
        model, metadata = load_model(artifact_dir=artifact_dir)
    frame = _records_to_frame(records)
    predictions = model.predict(frame)

    return {
        "model_name": metadata["model_name"],
        "predictions": [round(float(prediction), 2) for prediction in predictions],
        "records": len(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction locale a partir du modele entraine."
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Dossier contenant les artefacts du modele.",
    )
    parser.add_argument(
        "--payload",
        default=json.dumps(EXAMPLE_RECORD),
        help="Payload JSON representant un enregistrement d'entree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.payload)
    records = [payload] if isinstance(payload, dict) else payload
    result = predict_records(records=records, artifact_dir=args.artifact_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
