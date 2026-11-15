
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from ml_service import describe_churn_model


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "churn_model.joblib"
)


def save_churn_model(
    model: Pipeline,
    metrics: dict,
    train_size: int,
    test_size: int,
) -> dict:
    """Сохраняем модель и метаданные."""

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "pipeline": model,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "metrics": metrics,
        "train_size": train_size,
        "test_size": test_size,
        **describe_churn_model(model),
    }

    joblib.dump(model_bundle, MODEL_PATH)

    return model_bundle


def load_churn_model() -> dict | None:
    """Загружаем модель с диска."""

    if not MODEL_PATH.is_file():
        return None

    model_bundle = joblib.load(MODEL_PATH)

    required = (
        "trained_at",
        "metrics",
        "train_size",
        "test_size",
    )

    if (
        not isinstance(model_bundle, dict)
        or not isinstance(
            model_bundle.get("pipeline"),
            Pipeline,
        )
        or not all(
            key in model_bundle
            for key in required
        )
    ):
        raise ValueError(
            "Неверный формат файла модели"
        )

    # Поддержка моделей, сохранённых до Day 8
    model_info = describe_churn_model(
        model_bundle["pipeline"]
    )

    model_bundle.setdefault(
        "model_type",
        model_info["model_type"],
    )

    model_bundle.setdefault(
        "hyperparameters",
        model_info["hyperparameters"],
    )

    return model_bundle
