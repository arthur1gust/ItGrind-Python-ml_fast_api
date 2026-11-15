
import math

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from schemas import TrainingConfigChurn


# Параметры, разрешённые в учебном API
ALLOWED_HYPERPARAMETERS = {
    "logreg": {
        "C",
        "max_iter",
        "class_weight",
    },
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight",
    },
}


def create_classifier(config: TrainingConfigChurn):
    """Создаём классификатор по конфигурации."""

    model_type = config.model_type
    parameters = config.hyperparameters

    allowed = ALLOWED_HYPERPARAMETERS[model_type]

    invalid = set(parameters) - allowed

    if invalid:
        raise ValueError(
            f"Недопустимые параметры: {sorted(invalid)}. "
            f"Разрешены: {sorted(allowed)}"
        )

    # Проверяем веса классов
    if "class_weight" in parameters:
        value = parameters["class_weight"]

        if value not in (None, "balanced"):
            raise ValueError(
                "class_weight должен быть "
                "'balanced' или null"
            )

    # Проверяем целочисленные параметры
    integer_limits = {
        "max_iter": (1, 10000),
        "n_estimators": (1, 500),
        "max_depth": (1, 100),
        "min_samples_split": (2, 100000),
        "min_samples_leaf": (1, 100000),
    }

    for name, (minimum, maximum) in integer_limits.items():
        if name not in parameters:
            continue

        value = parameters[name]

        if name == "max_depth" and value is None:
            continue

        if (
            type(value) is not int
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"{name} должен быть целым числом "
                f"от {minimum} до {maximum}"
            )

    # Проверяем регуляризацию LogisticRegression
    if "C" in parameters:
        value = parameters["C"]

        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "C должен быть положительным числом"
            )

    # Выбираем алгоритм
    if model_type == "logreg":
        defaults = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        }

        return LogisticRegression(
            **{**defaults, **parameters}
        )

    defaults = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
    }

    return RandomForestClassifier(
        **{**defaults, **parameters}
    )


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfigChurn | None = None,
) -> Pipeline:
    """Обучаем выбранный классификатор."""

    if X_train.empty or y_train.empty:
        raise ValueError(
            "Обучающая выборка пуста"
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Для обучения нужны оба класса churn"
        )

    config = config or TrainingConfigChurn()

    classifier = create_classifier(config)

    model = Pipeline([
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "classifier",
            classifier,
        ),
    ])

    model.fit(X_train, y_train)

    return model


def describe_churn_model(model: Pipeline) -> dict:
    """Возвращаем тип и параметры модели."""

    classifier = model.named_steps["classifier"]

    if isinstance(classifier, LogisticRegression):
        model_type = "logreg"

        names = (
            "C",
            "max_iter",
            "class_weight",
            "random_state",
        )

    elif isinstance(classifier, RandomForestClassifier):
        model_type = "random_forest"

        names = (
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
            "random_state",
        )

    else:
        raise ValueError(
            "Неизвестный тип классификатора"
        )

    parameters = classifier.get_params()

    return {
        "model_type": model_type,
        "hyperparameters": {
            name: parameters[name]
            for name in names
        },
    }


def evaluate_churn_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Оцениваем качество модели."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
    }
