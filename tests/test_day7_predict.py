def test_predict_without_model(isolated_api, sample_client):
    response = isolated_api.post("/predict", json=sample_client)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_TRAINED"


def test_predict_invalid_request(isolated_api):
    response = isolated_api.post("/predict", json={"monthly_fee": "invalid"})
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_predict_single_and_batch(isolated_api, sample_client):
    trained = isolated_api.post("/model/train", json={
        "model_type": "logreg", "hyperparameters": {"max_iter": 300}
    })
    assert trained.status_code == 200, trained.text

    response = isolated_api.post("/predict", json=sample_client)
    assert response.status_code == 200, response.text
    item = response.json()
    assert item["churn"] in (0, 1)
    assert set(item["probabilities"]) == {"0", "1"}
    assert abs(sum(item["probabilities"].values()) - 1) < 1e-9

    batch = isolated_api.post("/predict", json=[sample_client, sample_client])
    assert batch.status_code == 200, batch.text
    assert batch.json() == [item, item]
