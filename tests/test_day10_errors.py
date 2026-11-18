import pytest
from fastapi.testclient import TestClient
from pandas.errors import EmptyDataError

import main
import model_storage
from main import app


VALID_CLIENT = {
    "monthly_fee": 49.99,
    "usage_hours": 12.5,
    "support_requests": 3,
    "account_age_months": 18,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Тесты не зависят от локального models/churn_model.joblib.
    monkeypatch.setattr(
        model_storage, "MODEL_PATH", tmp_path / "churn_model.joblib"
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def assert_error(response, http_status, error_code):
    assert response.status_code == http_status
    payload = response.json()
    assert set(payload) == {"code", "message", "details"}
    assert payload["code"] == error_code
    assert isinstance(payload["message"], str)
    assert isinstance(payload["details"], list)
    assert "Traceback" not in response.text
    return payload


def test_missing_required_feature(client):
    body = {k: v for k, v in VALID_CLIENT.items() if k != "monthly_fee"}
    result = assert_error(client.post("/predict", json=body), 422, "VALIDATION_ERROR")
    assert result["details"]
    assert any("monthly_fee" in e["field"] for e in result["details"])


def test_extra_feature_is_not_silently_ignored(client):
    body = {**VALID_CLIENT, "unknown_feature": 25}
    result = assert_error(client.post("/predict", json=body), 422, "VALIDATION_ERROR")
    assert any("unknown_feature" in e["field"] for e in result["details"])


def test_wrong_feature_type(client):
    body = {**VALID_CLIENT, "monthly_fee": "wrong"}
    result = assert_error(client.post("/predict", json=body), 422, "VALIDATION_ERROR")
    assert any("monthly_fee" in e["field"] for e in result["details"])


def test_empty_batch(client):
    assert_error(client.post("/predict", json=[]), 422, "EMPTY_BATCH")


def test_model_is_not_trained(client):
    assert_error(client.post("/predict", json=VALID_CLIENT), 503, "MODEL_NOT_TRAINED")


def test_dataset_missing(client, monkeypatch):
    def missing():
        raise FileNotFoundError("sensitive local path")
    monkeypatch.setattr(main, "load_dataset", missing)
    result = assert_error(
        client.post("/model/train", json={"model_type": "logreg"}),
        404, "DATASET_NOT_FOUND",
    )
    assert "sensitive local path" not in str(result)


def test_dataset_empty(client, monkeypatch):
    def empty():
        raise EmptyDataError("No columns to parse from file")
    monkeypatch.setattr(main, "load_dataset", empty)
    assert_error(
        client.post("/model/train", json={"model_type": "logreg"}),
        422, "DATASET_EMPTY",
    )


def test_invalid_hyperparameter(client):
    assert_error(
        client.post("/model/train", json={
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": -5},
        }),
        422, "INVALID_TRAINING_CONFIG",
    )


def test_unknown_url_is_unified(client):
    assert_error(client.get("/not-found-123"), 404, "NOT_FOUND")


def test_unexpected_exception_does_not_leak_traceback(client, monkeypatch):
    def crash():
        raise RuntimeError("PRIVATE_SERVER_SECRET")
    monkeypatch.setattr(main, "load_dataset", crash)
    result = assert_error(client.get("/dataset/info"), 500, "INTERNAL_ERROR")
    assert "PRIVATE_SERVER_SECRET" not in str(result)


def test_train_and_predict_remain_working(client):
    response = client.post("/model/train", json={
        "model_type": "logreg",
        "hyperparameters": {"max_iter": 1000},
    })
    assert response.status_code == 200, response.text
    result = client.post("/predict", json=VALID_CLIENT)
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["churn"] in (0, 1)
    assert abs(sum(payload["probabilities"].values()) - 1) < 1e-6


def test_openapi_error_examples(client):
    paths = client.get("/openapi.json").json()["paths"]
    train = paths["/model/train"]["post"]["responses"]
    predict = paths["/predict"]["post"]["responses"]
    for status in ("404", "422", "500"):
        assert status in train
    for status in ("422", "503", "500"):
        assert status in predict
    assert train["422"]["content"]["application/json"]["example"]["code"]
    assert predict["503"]["content"]["application/json"]["example"]["code"]
