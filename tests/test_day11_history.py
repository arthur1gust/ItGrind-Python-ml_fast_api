import json

import pytest
from fastapi.testclient import TestClient

import model_storage
import training_history
from main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Тесты не трогают рабочие модели и историю пользователя.
    monkeypatch.setattr(
        model_storage, "MODEL_PATH", tmp_path / "churn_model.joblib"
    )
    monkeypatch.setattr(
        training_history, "HISTORY_PATH", tmp_path / "training_history.json"
    )
    with TestClient(app, raise_server_exceptions=False) as api:
        yield api


def train(client, model_type="logreg", hyperparameters=None):
    return client.post(
        "/model/train",
        json={
            "model_type": model_type,
            "hyperparameters": hyperparameters or {},
        },
    )


def test_empty_history_before_first_training(client):
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "latest": None,
        "total": 0,
        "history": [],
    }


def test_training_writes_history_and_roc_auc(client):
    response = train(client, "logreg", {"C": 0.5})
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["roc_auc"] <= 1

    history_path = training_history.HISTORY_PATH
    assert history_path.is_file()
    raw_history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(raw_history) == 1

    result = client.get("/model/metrics")
    assert result.status_code == 200
    payload = result.json()
    assert payload["total"] == 1
    assert len(payload["history"]) == 1
    assert payload["latest"] == payload["history"][0]
    assert payload["latest"]["model_type"] == "logreg"
    assert payload["latest"]["hyperparameters"]["C"] == 0.5
    assert payload["latest"]["metrics"]["roc_auc"] == response.json()["roc_auc"]
    assert payload["latest"]["timestamp"] == response.json()["trained_at"]


def test_history_retains_both_models_and_filters(client):
    first = train(client, "logreg", {"max_iter": 200})
    second = train(client, "random_forest", {"n_estimators": 15, "max_depth": 4})
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    full = client.get("/model/metrics?limit=1").json()
    assert full["total"] == 2
    assert len(full["history"]) == 1
    assert full["latest"]["model_type"] == "random_forest"
    assert full["history"][0] == full["latest"]

    filtered = client.get("/model/metrics?model_type=logreg").json()
    assert filtered["total"] == 1
    assert filtered["latest"]["model_type"] == "logreg"
    assert filtered["history"][0]["model_type"] == "logreg"

    forest = client.get("/model/metrics?model_type=random_forest").json()
    assert forest["total"] == 1
    assert forest["latest"]["hyperparameters"]["n_estimators"] == 15


def test_history_survives_restart(client):
    response = train(client)
    assert response.status_code == 200, response.text
    saved = client.get("/model/metrics").json()

    # Симулируем перезапуск приложения: lifespan заново читает модель.
    with TestClient(app, raise_server_exceptions=False) as restarted:
        assert restarted.get("/model/status").json()["trained"] is True
        assert restarted.get("/model/metrics").json() == saved


def test_invalid_training_is_not_recorded(client):
    response = train(client, "random_forest", {"n_estimators": -5})
    assert response.status_code == 422
    assert client.get("/model/metrics").json()["total"] == 0
    assert not training_history.HISTORY_PATH.exists()


def test_query_parameters_are_validated(client):
    for path in (
        "/model/metrics?limit=0",
        "/model/metrics?limit=101",
        "/model/metrics?model_type=unknown",
    ):
        response = client.get(path)
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"


def test_corrupted_history_is_safe_500(client):
    training_history.HISTORY_PATH.write_text("invalid JSON", encoding="utf-8")
    response = client.get("/model/metrics")
    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "INTERNAL_ERROR"
    assert set(payload) == {"code", "message", "details"}
    assert "Traceback" not in response.text
