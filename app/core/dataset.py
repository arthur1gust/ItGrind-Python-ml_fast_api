"""Загрузка и базовая проверка входных CSV. Не зависит от FastAPI."""
from pathlib import Path
import pandas as pd
from app.ml.preprocessing import FEATURE_COLUMNS, prepare_data

def read_churn_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('Датасет пуст')
    missing = [field for field in (*FEATURE_COLUMNS, 'churn') if field not in df.columns]
    if missing:
        raise ValueError(f'Отсутствуют столбцы: {missing}')
    # Ошибки в признаках и целевой переменной выявляются до обучения.
    prepare_data(df)
    return df
