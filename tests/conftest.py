import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dataset_service
import model_storage
import training_history
from main import app


@pytest.fixture(autouse=True)
def isolate_training_history_for_all_tests(tmp_path, monkeypatch):
    """Даже старые тесты не должны писать в реальный JSON истории."""
    monkeypatch.setattr(
        training_history, "HISTORY_PATH", tmp_path / "training_history.json"
    )


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Детерминированный мини-датасет с двумя классами и всеми 9 признаками."""
    rows = []
    for i in range(60):
        rows.append({
            "monthly_fee": float(15 + (i * 7) % 75),
            "usage_hours": float(3 + (i * 11) % 40),
            "support_requests": int(i % 6),
            "account_age_months": int(1 + (i * 5) % 60),
            "failed_payments": int(i % 3),
            "region": ["europe", "asia", "america", "africa"][i % 4],
            "device_type": ["mobile", "desktop", "tablet"][i % 3],
            "payment_method": ["card", "paypal", "crypto"][i % 3],
            "autopay_enabled": int(i % 2),
            "churn": int(i % 4 == 0),
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "churn_dataset.csv"
    df.to_csv(path, index=False)
    return df, path


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch, synthetic_dataset):
    """Изолируем CSV, joblib и историю на протяжении одного теста."""
    df, csv_path = synthetic_dataset
    models_dir = tmp_path / "models"
    monkeypatch.setattr(dataset_service, "DATASET_PATH", csv_path)
    monkeypatch.setattr(model_storage, "MODEL_PATH", models_dir / "churn_model.joblib")
    monkeypatch.setattr(training_history, "HISTORY_PATH", models_dir / "training_history.json")
    return {"dataset": df, "csv": csv_path, "model": model_storage.MODEL_PATH,
            "history": training_history.HISTORY_PATH}


@pytest.fixture
def isolated_api(isolated_paths):
    """Контекстный TestClient запускает lifespan и загружает временную модель."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def sample_client(synthetic_dataset):
    df, _ = synthetic_dataset
    return df.drop(columns="churn").iloc[0].to_dict()
