from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from pandas.errors import EmptyDataError, ParserError

from dataset_service import load_dataset
from ml_service import evaluate_churn_model, train_churn_model
from model_storage import load_churn_model, save_churn_model
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    split_data,
)
from schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Восстанавливаем сохранённую модель при запуске приложения."""
    app.state.model_bundle = load_churn_model()
    yield
    app.state.model_bundle = None


app = FastAPI(
    title="ML Churn Prediction API",
    description="Service for customer churn prediction",
    version="0.8.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


def get_dataset() -> pd.DataFrame:
    """Загружаем CSV и преобразуем ошибки данных в HTTP-ошибки."""
    try:
        return load_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        ValueError,
        EmptyDataError,
        ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(default=5, ge=1, le=100)):
    df = get_dataset()
    preview = df.head(n)
    return (
        preview.astype(object)
        .where(preview.notna(), None)
        .to_dict(orient="records")
    )


@app.get("/dataset/info")
def dataset_info():
    df = get_dataset()
    counts = df["churn"].value_counts()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_names": list(FeatureVectorChurn.model_fields),
        "column_names": df.columns.tolist(),
        "churn_distribution": {
            "0": int(counts.get(0, 0)),
            "1": int(counts.get(1, 0)),
        },
    }


@app.get("/dataset/split-info")
def dataset_split_info():
    df = get_dataset()

    try:
        X_train, X_test, y_train, y_test = split_data(df)
        preprocessor = build_preprocessor()
        X_train_ready = preprocessor.fit_transform(X_train)
        preprocessor.transform(X_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "processed_features_count": int(X_train_ready.shape[1]),
        "train_churn_distribution": {
            "0": int(train_counts.get(0, 0)),
            "1": int(train_counts.get(1, 0)),
        },
        "test_churn_distribution": {
            "0": int(test_counts.get(0, 0)),
            "1": int(test_counts.get(1, 0)),
        },
    }


@app.post("/model/train")
def model_train(config: TrainingConfigChurn, request: Request):
    """Обучаем выбранный классификатор и сохраняем Pipeline."""
    df = get_dataset()

    try:
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_churn_model(X_train, y_train, config)
        metrics = evaluate_churn_model(model, X_test, y_test)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_bundle = save_churn_model(
        model=model,
        metrics=metrics,
        train_size=len(X_train),
        test_size=len(X_test),
    )
    request.app.state.model_bundle = model_bundle

    return {
        "message": "model trained successfully",
        "model": model.named_steps["classifier"].__class__.__name__,
        "model_type": model_bundle["model_type"],
        "hyperparameters": model_bundle["hyperparameters"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": model_bundle["trained_at"],
        **metrics,
    }


@app.get("/model/status")
def model_status(request: Request):
    """Возвращаем статус последней обученной модели."""
    bundle = getattr(request.app.state, "model_bundle", None)

    if bundle is None:
        return {
            "trained": False,
            "last_trained_at": None,
            "metrics": None,
            "model_type": None,
            "hyperparameters": None,
        }

    return {
        "trained": True,
        "last_trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
        "model_type": bundle["model_type"],
        "hyperparameters": bundle["hyperparameters"],
        "train_size": bundle["train_size"],
        "test_size": bundle["test_size"],
    }


EXAMPLE_CLIENT_1 = {
    "monthly_fee": 49.99,
    "usage_hours": 12.5,
    "support_requests": 3,
    "account_age_months": 18,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}

EXAMPLE_CLIENT_2 = {
    "monthly_fee": 9.99,
    "usage_hours": 35.0,
    "support_requests": 0,
    "account_age_months": 48,
    "failed_payments": 0,
    "region": "asia",
    "device_type": "desktop",
    "payment_method": "paypal",
    "autopay_enabled": 1,
}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Прогноз оттока клиентов",
    description=(
        "Принимает одного клиента или список клиентов. "
        "Возвращает класс churn и вероятности для классов 0 и 1. "
        "Модель должна быть предварительно обучена."
    ),
    responses={
        200: {
            "description": "Предсказание выполнено",
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Один клиент",
                            "value": {
                                "churn": 1,
                                "probabilities": {"0": 0.18, "1": 0.82},
                            },
                        },
                        "batch": {
                            "summary": "Два клиента",
                            "value": [
                                {
                                    "churn": 1,
                                    "probabilities": {"0": 0.18, "1": 0.82},
                                },
                                {
                                    "churn": 0,
                                    "probabilities": {"0": 0.77, "1": 0.23},
                                },
                            ],
                        },
                    }
                }
            },
        },
        503: {"description": "Модель ещё не обучена"},
    },
)
def predict(
    request: Request,
    features: Annotated[
        FeatureVectorChurn | list[FeatureVectorChurn],
        Body(
            openapi_examples={
                "single": {"summary": "Один клиент", "value": EXAMPLE_CLIENT_1},
                "batch": {
                    "summary": "Несколько клиентов",
                    "value": [EXAMPLE_CLIENT_1, EXAMPLE_CLIENT_2],
                },
            }
        ),
    ],
):
    """Предсказываем churn без переобучения модели."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не обучена. Сначала выполните POST /model/train",
        )

    is_batch = isinstance(features, list)
    clients = features if is_batch else [features]
    if not clients:
        raise HTTPException(
            status_code=422,
            detail="Список клиентов не должен быть пустым",
        )

    X = pd.DataFrame([client.model_dump() for client in clients])
    model = bundle["pipeline"]
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)

    classifier = model.named_steps["classifier"]
    class_indexes = {
        int(label): index
        for index, label in enumerate(classifier.classes_)
    }

    results = [
        PredictionResponseChurn(
            churn=int(predicted[i]),
            probabilities={
                "0": float(probabilities[i, class_indexes[0]]),
                "1": float(probabilities[i, class_indexes[1]]),
            },
        )
        for i in range(len(clients))
    ]

    return results if is_batch else results[0]
