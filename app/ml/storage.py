"""Persist a fitted Pipeline with its training metadata."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_bundle(
    model: Pipeline,
    metrics: dict,
    train_size: int,
    test_size: int,
    path: Path,
) -> dict:
    classifier = model.named_steps["classifier"]
    classifier_name = type(classifier).__name__
    model_types = {
        "LogisticRegression": "logreg",
        "RandomForestClassifier": "random_forest",
    }
    if classifier_name not in model_types:
        raise ValueError(f"Неподдерживаемый классификатор: {classifier_name}")

    bundle = {
        "pipeline": model,
        "model_type": model_types[classifier_name],
        "hyperparameters": classifier.get_params(deep=False),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": int(train_size),
        "test_size": int(test_size),
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    # Атомарно заменяем модель, чтобы ошибка записи не повредила старый файл.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".churn_model_",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)
        joblib.dump(bundle, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return bundle


def load_bundle(path: Path) -> dict | None:
    if not path.is_file():
        return None
    # Внимание: joblib/pickle загружать только из доверенных источников.
    return joblib.load(path)
