"""HTTP-маршруты churn API; ML-компоненты и хранилища в отдельных пакетах."""
from time import perf_counter
from typing import Annotated, Callable, Literal

import dataset_service
import pandas as pd
from fastapi import APIRouter, Body, Query, Request
from sklearn.metrics import roc_auc_score

from app.core.errors import PREDICT_ERROR_RESPONSES, TRAIN_ERROR_RESPONSES, ServiceError
from app.ml.preprocessing import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURE_COLUMNS,
    prepare_prediction_data, build_preprocessor, split_data,
)
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from ml_service import evaluate_churn_model, train_churn_model
from model_storage import save_churn_model
from monitoring import configure_logging
from training_history import append_training_record, get_training_metrics

logger = configure_logging()


def create_router(dataset_loader: Callable[[], pd.DataFrame]) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def root():
        return {"message": "ml churn service is running"}


    @router.get(
        "/health",
        summary="Состояние churn сервиса",
        description=(
            "Сообщает, удалось ли загрузить CSV и доступна ли обученная модель. "
            "Отсутствие модели до первого обучения не мешает работе API. "
            "HTTP 200 означает, что веб-сервис отвечает, даже если status=degraded."
        ),
    )
    def health(request: Request):
        """Лёгкий диагностический endpoint без переобучения и повторного чтения CSV."""
        dataset_loaded = bool(getattr(request.app.state, "dataset_loaded", False))
        # Если CSV удалили после старта, это тоже отражается в ответе.
        dataset_loaded = dataset_loaded and dataset_service.DATASET_PATH.is_file()
        model_available = getattr(request.app.state, "model_bundle", None) is not None
        return {
            "status": "ok" if dataset_loaded else "degraded",
            "model_available": model_available,
            "dataset_loaded": dataset_loaded,
        }


    @router.get("/dataset/preview")
    def dataset_preview(n: int = Query(default=5, ge=1, le=100)):
        df = dataset_loader()
        preview = df.head(n)
        return (
            preview.astype(object)
            .where(preview.notna(), None)
            .to_dict(orient="records")
        )


    @router.get("/dataset/info")
    def dataset_info():
        df = dataset_loader()
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


    @router.get("/dataset/split-info")
    def dataset_split_info():
        df = dataset_loader()

        try:
            X_train, X_test, y_train, y_test = split_data(df)
            preprocessor = build_preprocessor()
            X_train_ready = preprocessor.fit_transform(X_train)
            preprocessor.transform(X_test)
        except ValueError as exc:
            raise ServiceError(
                "DATASET_INVALID", "Не удалось подготовить обучающую выборку", 422,
                details=[{"reason": str(exc)}],
            ) from exc

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


    @router.post("/model/train", responses=TRAIN_ERROR_RESPONSES)
    def model_train(config: TrainingConfigChurn, request: Request):
        """Обучаем выбранный классификатор и сохраняем Pipeline."""
        started = perf_counter()
        logger.info("event=training_started model_type=%s", config.model_type)
        df = dataset_loader()

        try:
            X_train, X_test, y_train, y_test = split_data(df)
            model = train_churn_model(X_train, y_train, config)
            metrics = evaluate_churn_model(model, X_test, y_test)

            # Дополнительная метрика: качество ранжирования уходящих клиентов.
            if y_test.nunique() == 2:
                classes = list(model.named_steps["classifier"].classes_)
                positive_index = classes.index(1)
                scores = model.predict_proba(X_test)[:, positive_index]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, scores)), 4)
        except (ValueError, TypeError) as exc:
            raise ServiceError(
                "INVALID_TRAINING_CONFIG",
                "Некорректные параметры обучения или недостаточно данных",
                422,
                details=[{"reason": str(exc)}],
            ) from exc

        model_bundle = save_churn_model(
            model=model,
            metrics=metrics,
            train_size=len(X_train),
            test_size=len(X_test),
        )
        # Добавляем запись только после успешного обучения и сохранения модели.
        append_training_record(model_bundle)
        request.app.state.model_bundle = model_bundle
        logger.info(
            "event=training_finished model_type=%s train_size=%d test_size=%d "
            "accuracy=%.4f f1=%.4f elapsed_ms=%.1f",
            model_bundle["model_type"], len(X_train), len(X_test),
            metrics["accuracy"], metrics["f1"], (perf_counter() - started) * 1000,
        )

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


    @router.get("/model/status")
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


    @router.get(
        "/model/metrics",
        summary="Метрики и история обучений",
        description=(
            "Последние результаты обучения. Опционально фильтрует историю "
            "по типу модели и ограничивает количество записей. "
            "История хранится на диске между перезапусками сервера."
        ),
    )
    def model_metrics(
        limit: int = Query(default=5, ge=1, le=100),
        model_type: Literal["logreg", "random_forest"] | None = Query(default=None),
    ):
        return get_training_metrics(limit=limit, model_type=model_type)


    @router.get("/model/schema")
    def model_schema():
        """Описываем обязательные признаки и их типы для POST /predict."""
        return {
            "feature_names": FEATURE_COLUMNS,
            "features": [
                {
                    "name": name,
                    "type": FeatureVectorChurn.model_fields[name].annotation.__name__,
                    "category": (
                        "numeric" if name in NUMERIC_FEATURES else "categorical"
                    ),
                    "required": FeatureVectorChurn.model_fields[name].is_required(),
                }
                for name in FEATURE_COLUMNS
            ],
            "target": {
                "name": "churn",
                "type": "int",
                "values": [0, 1],
            },
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


    @router.post(
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
            **PREDICT_ERROR_RESPONSES,
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
        started = perf_counter()
        is_batch = isinstance(features, list)
        clients = features if is_batch else [features]
        if not clients:
            raise ServiceError(
                "EMPTY_BATCH", "Список клиентов не должен быть пустым", 422
            )

        bundle = getattr(request.app.state, "model_bundle", None)
        if bundle is None:
            raise ServiceError(
                "MODEL_NOT_TRAINED",
                "Модель не обучена. Сначала выполните POST /model/train",
                503,
            )

        try:
            X = prepare_prediction_data(clients)
        except (ValueError, TypeError) as exc:
            raise ServiceError(
                "INVALID_FEATURES", "Некорректные признаки клиента", 422,
                details=[{"reason": str(exc)}],
            ) from exc

        model = bundle["pipeline"]
        try:
            predicted = model.predict(X)
            probabilities = model.predict_proba(X)
        except (ValueError, TypeError) as exc:
            # Некорректные данные не приводят к технической трассировке в HTTP-ответе.
            raise ServiceError(
                "PREDICTION_FAILED", "Не удалось выполнить прогноз", 422,
            ) from exc

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

        logger.info(
            "event=prediction_finished model_type=%s batch_size=%d elapsed_ms=%.1f",
            bundle["model_type"], len(clients), (perf_counter() - started) * 1000,
        )
        return results if is_batch else results[0]

    return router
