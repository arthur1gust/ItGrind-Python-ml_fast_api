
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def prepare_data(df: pd.DataFrame):
    """Отделяем признаки X от целевой переменной y."""

    required = FEATURE_COLUMNS + ["churn"]

    missing = [
        name for name in required
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Отсутствуют столбцы: {missing}"
        )

    if df.empty:
        raise ValueError("Датасет пуст")

    if df["churn"].isna().any():
        raise ValueError(
            "Пропуски в churn недопустимы"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df["churn"], errors="raise")

    if not y.isin([0, 1]).all():
        raise ValueError(
            "churn должен содержать только 0 и 1"
        )

    y = y.astype(int)

    # Приводим числовые признаки к числам
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column], errors="raise"
        )

    # Приводим пропуски категорий к np.nan
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(object)
        X.loc[X[column].isna(), column] = np.nan

    return X, y


def split_data(df: pd.DataFrame):
    """Разделяем данные на train и test."""

    X, y = prepare_data(df)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_preprocessor():
    """Создаём правила предобработки признаков."""

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )
