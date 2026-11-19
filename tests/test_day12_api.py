"""Day 12: сквозные API-тесты CSV -> train -> status -> predict."""

from fastapi.testclient import TestClient

import dataset_service
import model_storage
import training_history
from main import app
from preprocessing import FEATURE_COLUMNS


def assert_error(response, status_code, code):
    assert response.status_code == status_code, response.text
    data = response.json()
    assert set(data) == {"code", "message", "details"}
    assert data["code"] == code
    assert "Traceback" not in response.text


def test_dataset_endpoints_use_synthetic_csv(isolated_api):
    preview = isolated_api.get("/dataset/preview?n=3")
    assert preview.status_code == 200, preview.text
    assert len(preview.json()) == 3
    assert set(preview.json()[0]) == set(FEATURE_COLUMNS + ["churn"])

    info = isolated_api.get("/dataset/info")
    assert info.status_code == 200, info.text
    assert info.json()["rows"] == 60
    assert info.json()["columns"] == 10
    assert info.json()["churn_distribution"] == {"0": 45, "1": 15}

    split = isolated_api.get("/dataset/split-info")
    assert split.status_code == 200, split.text
    assert (split.json()["train_size"], split.json()["test_size"]) == (48, 12)


def test_full_api_flow_from_training_to_prediction(isolated_api, isolated_paths, sample_client):
    assert isolated_api.get("/model/status").json()["trained"] is False
    assert isolated_api.get("/model/metrics").json()["total"] == 0

    trained = isolated_api.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {"max_iter": 300}},
    )
    assert trained.status_code == 200, trained.text
    assert trained.json()["model_type"] == "logreg"
    assert 0 <= trained.json()["accuracy"] <= 1
    assert 0 <= trained.json()["f1"] <= 1
    assert 0 <= trained.json()["roc_auc"] <= 1
    assert isolated_paths["model"].is_file()
    assert isolated_paths["history"].is_file()

    status = isolated_api.get("/model/status")
    assert status.status_code == 200
    assert status.json()["trained"] is True
    assert status.json()["model_type"] == "logreg"

    single = isolated_api.post("/predict", json=sample_client)
    assert single.status_code == 200, single.text
    result = single.json()
    assert result["churn"] in (0, 1)
    assert set(result["probabilities"]) == {"0", "1"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-9

    batch = isolated_api.post("/predict", json=[sample_client, sample_client])
    assert batch.status_code == 200, batch.text
    assert batch.json() == [result, result]

    history = isolated_api.get("/model/metrics")
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["latest"]["model_type"] == "logreg"


def test_restart_restores_model_and_history(isolated_paths, sample_client):
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/model/train", json={
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 12, "max_depth": 4},
        })
        assert response.status_code == 200, response.text
        old_status = client.get("/model/status").json()
        old_prediction = client.post("/predict", json=sample_client).json()

    # Новый lifespan загружает joblib с диска. Переобучения нет.
    with TestClient(app, raise_server_exceptions=False) as restarted:
        assert restarted.get("/model/status").json() == old_status
        assert restarted.post("/predict", json=sample_client).json() == old_prediction
        assert restarted.get("/model/metrics").json()["total"] == 1


def test_retraining_with_same_config_is_reproducible(isolated_api, sample_client):
    config = {"model_type": "logreg", "hyperparameters": {"max_iter": 300}}
    first = isolated_api.post("/model/train", json=config)
    assert first.status_code == 200, first.text
    prediction_before = isolated_api.post("/predict", json=sample_client).json()
    second = isolated_api.post("/model/train", json=config)
    assert second.status_code == 200, second.text
    prediction_after = isolated_api.post("/predict", json=sample_client).json()
    for key in ("accuracy", "f1", "roc_auc"):
        assert first.json()[key] == second.json()[key]
    assert prediction_before == prediction_after
    assert isolated_api.get("/model/metrics").json()["total"] == 2


def test_predict_without_trained_model(isolated_api, sample_client):
    assert_error(
        isolated_api.post("/predict", json=sample_client),
        503, "MODEL_NOT_TRAINED",
    )


def test_invalid_feature_data(isolated_api, sample_client):
    bad = {**sample_client, "monthly_fee": "not_a_number"}
    assert_error(isolated_api.post("/predict", json=bad), 422, "VALIDATION_ERROR")
    missing = {k: v for k, v in sample_client.items() if k != "region"}
    assert_error(isolated_api.post("/predict", json=missing), 422, "VALIDATION_ERROR")


def test_missing_csv_returns_404(isolated_api, isolated_paths, monkeypatch):
    missing_path = isolated_paths["csv"].with_name("missing.csv")
    monkeypatch.setattr(dataset_service, "DATASET_PATH", missing_path)
    assert_error(isolated_api.post("/model/train", json={}), 404, "DATASET_NOT_FOUND")


def test_empty_csv_returns_422(isolated_api, isolated_paths):
    isolated_paths["csv"].write_text("", encoding="utf-8")
    assert_error(isolated_api.post("/model/train", json={}), 422, "DATASET_EMPTY")


def test_invalid_train_hyperparameter_returns_422(isolated_api, isolated_paths):
    invalid = isolated_api.post("/model/train", json={
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": -5},
    })
    assert_error(invalid, 422, "INVALID_TRAINING_CONFIG")
    assert not isolated_paths["model"].exists()
    assert not isolated_paths["history"].exists()


def test_model_schema_matches_prediction_contract(isolated_api):
    response = isolated_api.get("/model/schema")
    assert response.status_code == 200
    payload = response.json()
    assert payload["feature_names"] == FEATURE_COLUMNS
    assert len(payload["features"]) == 9
    assert payload["target"]["name"] == "churn"
