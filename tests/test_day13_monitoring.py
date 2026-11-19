"""Day 13: мониторинг, /health, совместимость API и Docker-контракт."""

import logging
from pathlib import Path

from fastapi.testclient import TestClient

import dataset_service
from main import app


class CollectLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_health_before_training(isolated_api):
    result = isolated_api.get("/health")
    assert result.status_code == 200
    assert result.json() == {
        "status": "ok", "model_available": False, "dataset_loaded": True
    }


def test_health_after_training(isolated_api):
    trained = isolated_api.post(
        "/model/train",
        json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}},
    )
    assert trained.status_code == 200, trained.text
    assert isolated_api.get("/health").json() == {
        "status": "ok", "model_available": True, "dataset_loaded": True
    }


def test_health_missing_dataset_does_not_break_service(isolated_paths, monkeypatch):
    missing = isolated_paths["csv"].with_name("no-such-file.csv")
    monkeypatch.setattr(dataset_service, "DATASET_PATH", missing)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").json() == {
            "status": "degraded", "model_available": False, "dataset_loaded": False
        }
        assert client.get("/docs").status_code == 200
        assert client.get("/dataset/info").status_code == 404


def test_health_empty_dataset_and_errors(isolated_paths):
    isolated_paths["csv"].write_text("", encoding="utf-8")
    with TestClient(app, raise_server_exceptions=False) as client:
        status = client.get("/health")
        assert status.json()["dataset_loaded"] is False
        invalid = client.post("/model/train", json={"model_type": "logreg"})
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "DATASET_EMPTY"


def test_health_detects_deleted_csv(isolated_api, isolated_paths):
    assert isolated_api.get("/health").json()["dataset_loaded"] is True
    isolated_paths["csv"].unlink()
    assert isolated_api.get("/health").json()["dataset_loaded"] is False


def test_health_does_not_retrain_or_reload_dataset(isolated_api, monkeypatch):
    def no_loading():
        raise AssertionError("/health must not read CSV")

    monkeypatch.setattr("main.load_dataset", no_loading)
    assert isolated_api.get("/health").status_code == 200


def test_events_logged_without_client_values(isolated_api, sample_client):
    sink = CollectLogs()
    logger = logging.getLogger("churn_service")
    logger.addHandler(sink)
    try:
        result = isolated_api.post("/model/train", json={"model_type": "logreg"})
        assert result.status_code == 200, result.text
        predicted = isolated_api.post("/predict", json=sample_client)
        assert predicted.status_code == 200, predicted.text
        invalid = isolated_api.post("/predict", json={"monthly_fee": "secret-CLIENT-INPUT"})
        assert invalid.status_code == 422
    finally:
        logger.removeHandler(sink)
    joined = "\n".join(sink.messages)
    assert "event=dataset_loaded" in joined
    assert "event=training_started" in joined
    assert "event=training_finished" in joined
    assert "event=prediction_finished" in joined
    assert "event=request_validation_failed" in joined
    assert "secret-CLIENT-INPUT" not in joined


def test_service_errors_and_unhandled_errors_are_logged(isolated_api, monkeypatch):
    sink = CollectLogs()
    logger = logging.getLogger("churn_service")
    logger.addHandler(sink)
    try:
        not_found = isolated_api.get("/does-not-exist")
        assert not_found.status_code == 404
        assert not_found.json()["code"] == "NOT_FOUND"

        missing_model = isolated_api.post("/predict", json={
            "monthly_fee": 10.0, "usage_hours": 20.0, "support_requests": 0,
            "account_age_months": 4, "failed_payments": 0,
            "region": "asia", "device_type": "mobile",
            "payment_method": "card", "autopay_enabled": 1,
        })
        assert missing_model.status_code == 503

        def fail():
            raise RuntimeError("INTERNAL-SENSITIVE-STRING")

        monkeypatch.setattr("main.load_dataset", fail)
        response = isolated_api.get("/dataset/info")
        assert response.status_code == 500
        assert response.json() == {
            "code": "INTERNAL_ERROR", "message": "Внутренняя ошибка сервера", "details": []
        }
        assert "INTERNAL-SENSITIVE-STRING" not in response.text
    finally:
        logger.removeHandler(sink)
    joined = "\n".join(sink.messages)
    assert "event=http_error" in joined
    assert "event=service_error" in joined
    assert "event=unexpected_error" in joined


def test_openapi_and_health_docs(isolated_api):
    assert isolated_api.get("/docs").status_code == 200
    schema = isolated_api.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert schema["paths"]["/health"]["get"]["summary"]


def test_docker_contract_files():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    assert "python:3.13-slim" in dockerfile
    assert "uvicorn" in dockerfile
    assert "main:app" in dockerfile
    assert "0.0.0.0" in dockerfile
    assert "/health" in dockerfile
    assert "models/" in dockerignore
    assert ".venv" in dockerignore
    assert (root / "requirements-docker.txt").is_file()


def test_docker_compose_mounts_dataset_and_persists_models():
    """Данные excluded из image, значит compose обязан монтировать CSV с хоста."""
    root = Path(__file__).resolve().parents[1]
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    ignore = (root / ".dockerignore").read_text(encoding="utf-8")
    assert "churn_dataset.csv" in ignore
    assert "./data:/app/data:ro" in compose
    assert "churn-models:/app/models" in compose
    assert "8000:8000" in compose
