
import pytest
from fastapi.testclient import TestClient

from main import app
from model_storage import load_churn_model
import model_storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    temporary_path = (
        tmp_path / "churn_model.joblib"
    )

    monkeypatch.setattr(
        model_storage,
        "MODEL_PATH",
        temporary_path,
    )

    with TestClient(app) as test_client:
        yield test_client


def test_train_logreg(client):
    response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {
                "C": 0.5,
                "max_iter": 1500,
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "logreg"
    assert data["hyperparameters"]["C"] == 0.5
    assert "accuracy" in data
    assert "f1" in data


def test_train_random_forest(client):
    response = client.post(
        "/model/train",
        json={
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 30,
                "max_depth": 5,
            },
        },
    )

    assert response.status_code == 200

    status = client.get("/model/status").json()

    assert status["trained"] is True
    assert status["model_type"] == "random_forest"

    saved = load_churn_model()

    assert saved is not None
    assert saved["model_type"] == "random_forest"
    assert saved["hyperparameters"]["n_estimators"] == 30


def test_invalid_hyperparameter(client):
    response = client.post(
        "/model/train",
        json={
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": -5,
            },
        },
    )

    assert response.status_code == 422
