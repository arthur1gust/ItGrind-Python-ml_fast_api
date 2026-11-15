
from fastapi.testclient import TestClient

from main import app


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


def test_predict_without_model():
    with TestClient(app) as client:
        app.state.model_bundle = None

        response = client.post(
            "/predict",
            json=CLIENT,
        )

        assert response.status_code == 503


def test_predict_invalid_request():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"monthly_fee": "invalid"},
        )

        assert response.status_code == 422


def test_predict_single_and_batch():
    with TestClient(app) as client:
        # Для этого теста нужна сохранённая модель
        assert app.state.model_bundle is not None

        response = client.post(
            "/predict",
            json=CLIENT,
        )

        assert response.status_code == 200

        result = response.json()

        assert result["churn"] in (0, 1)
        assert set(
            result["probabilities"]
        ) == {"0", "1"}

        assert abs(
            sum(result["probabilities"].values()) - 1
        ) < 1e-6

        response = client.post(
            "/predict",
            json=[CLIENT, CLIENT],
        )

        assert response.status_code == 200
        assert len(response.json()) == 2
