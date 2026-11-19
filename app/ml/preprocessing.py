"""Общий контракт признаков для обучения и предсказания churn."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.churn import FeatureVectorChurn


NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяем набор полей и приводим их к общему порядку и типам."""
    missing = [name for name in FEATURE_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют признаки: {missing}")

    X = df.loc[:, FEATURE_COLUMNS].copy()

    for name in NUMERIC_FEATURES:
        X[name] = pd.to_numeric(X[name], errors="raise")

    for name in CATEGORICAL_FEATURES:
        X[name] = X[name].astype(object)
        X[name] = X[name].where(X[name].notna(), np.nan)

    return X


def prepare_data(df: pd.DataFrame):
    """Разделяем табличные данные на X и y, не обучая преобразования."""
    if df.empty:
        raise ValueError("Датасет пуст")
    if "churn" not in df.columns:
        raise ValueError("Отсутствует столбец churn")
    if df["churn"].isna().any():
        raise ValueError("Пропуски в churn недопустимы")

    X = prepare_feature_dataframe(df)
    y = pd.to_numeric(df["churn"], errors="raise")
    if not y.isin([0, 1]).all():
        raise ValueError("churn должен содержать только 0 и 1")

    return X, y.astype(int)


def prepare_prediction_data(
    features: FeatureVectorChurn | list[FeatureVectorChurn],
) -> pd.DataFrame:
    """Преобразуем один или несколько проверенных объектов в X для Pipeline."""
    clients = features if isinstance(features, list) else [features]
    if not clients:
        raise ValueError("Список клиентов не должен быть пустым")

    records = [client.model_dump() for client in clients]
    return prepare_feature_dataframe(pd.DataFrame.from_records(records))


def split_data(df: pd.DataFrame):
    X, y = prepare_data(df)
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
