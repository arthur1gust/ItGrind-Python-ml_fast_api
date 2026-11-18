import numpy as np
from fastapi.testclient import TestClient

import model_storage
from dataset_service import load_dataset
from main import app
from ml_service import train_churn_model
from model_storage import load_churn_model, save_churn_model
from preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    prepare_data,
    prepare_prediction_data,
    split_data,
)
from schemas import FeatureVectorChurn, TrainingConfigChurn


CLIENT = {
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


def test_feature_columns_match_training_and_predict():
    df = load_dataset()
    X_train, y = prepare_data(df)

    # Имитация JSON с изменённым порядком ключей.
    shuffled_client = FeatureVectorChurn.model_validate(
        dict(reversed(list(CLIENT.items())))
    )
    X_predict = prepare_prediction_data(shuffled_client)

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_predict.columns) == FEATURE_COLUMNS
    assert X_predict.shape == (1, 9)
    assert X_predict["region"].iloc[0] == "europe"


def test_model_schema_endpoint():
    with TestClient(app) as client:
        response = client.get("/model/schema")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_names"] == FEATURE_COLUMNS
    types = {item["name"]: item["type"] for item in data["features"]}
    assert types["monthly_fee"] == "float"
    assert types["region"] == "str"
    assert types["autopay_enabled"] == "int"
    assert all(item["required"] for item in data["features"])


def test_whole_pipeline_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_storage, "MODEL_PATH", tmp_path / "churn_model.joblib"
    )
    X_train, X_test, y_train, y_test = split_data(load_dataset())
    model = train_churn_model(
        X_train, y_train,
        TrainingConfigChurn(model_type="logreg", hyperparameters={"max_iter": 500}),
    )
    bundle = save_churn_model(
        model, {"accuracy": 0.0, "f1": 0.0}, len(X_train), len(X_test)
    )
    restored = load_churn_model()

    assert restored is not None
    assert restored["pipeline"].named_steps.keys() == model.named_steps.keys()
    X_predict = prepare_prediction_data(FeatureVectorChurn(**CLIENT))
    np.testing.assert_allclose(
        model.predict_proba(X_predict),
        restored["pipeline"].predict_proba(X_predict),
        rtol=0, atol=1e-12,
    )
    assert bundle["model_type"] == restored["model_type"]
