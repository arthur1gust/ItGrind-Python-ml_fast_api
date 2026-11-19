"""Day 12: unit-тесты подготовки данных и обучения без HTTP/FastAPI."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import dataset_service
import model_storage
from ml_service import create_classifier, evaluate_churn_model, train_churn_model
from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    prepare_data,
    prepare_prediction_data,
    split_data,
)
from schemas import FeatureVectorChurn, TrainingConfigChurn


def test_csv_loads_into_validated_dataframe(isolated_paths):
    df = dataset_service.load_dataset()
    assert df.shape == (60, 10)
    assert set(df.columns) == set(FEATURE_COLUMNS + ["churn"])
    assert set(df["churn"]) == {0, 1}


def test_prepare_data_separates_x_y_and_restores_column_order(synthetic_dataset):
    df, _ = synthetic_dataset
    shuffled = df[list(reversed(df.columns))]
    X, y = prepare_data(shuffled)
    assert X.shape == (60, 9)
    assert y.shape == (60,)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "churn" not in X
    assert set(y.unique()) == {0, 1}


def test_prepare_data_requires_target(synthetic_dataset):
    df, _ = synthetic_dataset
    with pytest.raises(ValueError, match="churn"):
        prepare_data(df.drop(columns="churn"))


def test_split_is_reproducible_and_stratified(synthetic_dataset):
    df, _ = synthetic_dataset
    first = split_data(df)
    second = split_data(df)
    for a, b in zip(first, second):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        else:
            pd.testing.assert_series_equal(a, b)
    X_train, X_test, y_train, y_test = first
    assert (len(X_train), len(X_test)) == (48, 12)
    assert abs(float(y_train.mean()) - float(y_test.mean())) <= 0.05
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_preprocessor_imputes_and_encodes(synthetic_dataset):
    df, _ = synthetic_dataset
    X, _ = prepare_data(df)
    X_train = X.iloc[:48].copy()
    X_test = X.iloc[48:].copy()
    X_train.loc[X_train.index[0], "monthly_fee"] = np.nan
    X_train.loc[X_train.index[1], "region"] = np.nan
    # Неизвестную категорию OneHotEncoder должен игнорировать.
    X_test.loc[X_test.index[0], "region"] = "new_region"

    preprocessor = build_preprocessor()
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    for matrix in (train_matrix, test_matrix):
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        assert np.isfinite(dense).all()
    assert train_matrix.shape[0] == 48
    assert test_matrix.shape[1] == train_matrix.shape[1]
    assert len(NUMERIC_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 3


@pytest.mark.parametrize(
    "model_type,hyperparameters",
    [
        ("logreg", {"max_iter": 300, "C": 0.5}),
        ("random_forest", {"n_estimators": 12, "max_depth": 4}),
    ],
)
def test_training_pipeline_and_metrics(synthetic_dataset, model_type, hyperparameters):
    df, _ = synthetic_dataset
    X_train, X_test, y_train, y_test = split_data(df)
    config = TrainingConfigChurn(
        model_type=model_type, hyperparameters=hyperparameters
    )
    model = train_churn_model(X_train, y_train, config)
    assert isinstance(model, Pipeline)
    assert set(model.named_steps) == {"preprocessor", "classifier"}
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(X_test), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    metrics = evaluate_churn_model(model, X_test, y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_prediction_dataframe_matches_training_feature_order(synthetic_dataset):
    df, _ = synthetic_dataset
    reversed_fields = dict(reversed(list(df.drop(columns="churn").iloc[0].to_dict().items())))
    model_input = FeatureVectorChurn.model_validate(reversed_fields)
    X = prepare_prediction_data(model_input)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (1, 9)


def test_joblib_roundtrip_preserves_predictions(isolated_paths):
    df = dataset_service.load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train_churn_model(X_train, y_train, TrainingConfigChurn())
    metrics = evaluate_churn_model(pipeline, X_test, y_test)
    model_storage.save_churn_model(pipeline, metrics, len(X_train), len(X_test))
    restored = model_storage.load_churn_model()
    assert model_storage.MODEL_PATH.exists()
    assert restored is not None
    assert restored["model_type"] == "logreg"
    np.testing.assert_array_equal(
        pipeline.predict(X_test), restored["pipeline"].predict(X_test)
    )
    np.testing.assert_allclose(
        pipeline.predict_proba(X_test),
        restored["pipeline"].predict_proba(X_test),
        rtol=0, atol=1e-12,
    )


def test_invalid_hyperparameter_is_rejected():
    config = TrainingConfigChurn(
        model_type="random_forest", hyperparameters={"n_estimators": -1}
    )
    with pytest.raises(ValueError, match="n_estimators"):
        create_classifier(config)
