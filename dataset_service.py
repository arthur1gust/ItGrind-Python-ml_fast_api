
from pathlib import Path

import pandas as pd
from pydantic import ValidationError, create_model

from schemas import (
    FeatureVectorChurn,
    DatasetRowChurn,
)


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "churn_dataset.csv"
)


# Вспомогательная схема:
# допускаем пропуски только в признаках.
NullableDatasetRowChurn = create_model(
    "NullableDatasetRowChurn",
    **{
        name: (
            (int, ...)
            if name == "churn"
            else (field.annotation | None, None)
        )
        for name, field in
        DatasetRowChurn.model_fields.items()
    }
)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}"
        )

    raw_df = pd.read_csv(DATASET_PATH)

    required_columns = list(
        DatasetRowChurn.model_fields
    )

    missing_columns = [
        name for name in required_columns
        if name not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Отсутствуют столбцы: {missing_columns}"
        )

    if raw_df.empty:
        raise ValueError("Датасет пуст")

    validated_rows = []

    for line_number, values in enumerate(
        raw_df[required_columns].to_dict(
            orient="records"
        ),
        start=2,
    ):
        payload = {
            name: (
                None if pd.isna(value) else value
            )
            for name, value in values.items()
        }

        has_missing_features = any(
            payload[name] is None
            for name in FeatureVectorChurn.model_fields
        )

        try:
            if has_missing_features:
                row = NullableDatasetRowChurn.model_validate(
                    payload
                )
            else:
                row = DatasetRowChurn.model_validate(
                    payload
                )

        except ValidationError as exc:
            raise ValueError(
                f"Ошибка в строке CSV "
                f"{line_number}: {exc}"
            ) from exc

        if row.churn not in (0, 1):
            raise ValueError(
                f"Строка {line_number}: "
                "churn должен быть 0 или 1"
            )

        validated_rows.append(row.model_dump())

    return pd.DataFrame(
        validated_rows,
        columns=required_columns,
    )
