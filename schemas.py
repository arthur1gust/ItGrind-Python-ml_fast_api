from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    # Лишние признаки — ошибка 422, а не незаметное игнорирование.
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: Literal[0, 1] = Field(
        description="Предсказанный класс: 0 — останется, 1 — уйдёт"
    )
    probabilities: dict[str, float] = Field(
        description="Вероятности классов 0 и 1"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"churn": 1, "probabilities": {"0": 0.18, "1": 0.82}}
            ]
        }
    }


class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"] = Field(
        default="logreg", description="Тип ML-модели"
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Гиперпараметры классификатора"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "random_forest",
                    "hyperparameters": {"n_estimators": 150, "max_depth": 8},
                }
            ]
        }
    }
