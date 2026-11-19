"""Day 14: smoke/regression checks of release layout, API and documentation."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.core.config import BASE_DIR
from app.core.dataset import read_churn_csv
from app.ml.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from app.ml.service import create_classifier
from app.schemas.churn import FeatureVectorChurn, TrainingConfigChurn
from main import app


def test_packages_are_real_modules():
    from app.api.routes import create_router
    from app.ml.storage import save_bundle, load_bundle
    from app.ml.history import get_training_metrics
    from app.core.errors import ServiceError
    assert callable(create_router)
    assert callable(save_bundle) and callable(load_bundle)
    assert callable(get_training_metrics)
    assert issubclass(ServiceError, Exception)


def test_final_release_has_documentation_and_docker_files():
    required = ("README.md", "requirements.txt", "requirements-dev.txt", "Dockerfile",
                ".dockerignore", ".gitignore", "pytest.ini", "compose.yaml")
    assert all((BASE_DIR / f).is_file() for f in required)
    doc = (BASE_DIR / 'README.md').read_text(encoding='utf-8')
    for term in ('/model/train', '/predict', '/health', 'docker build', 'churn_dataset.csv'):
        assert term in doc


def test_demo_csv_matches_feature_contract():
    frame = read_churn_csv(BASE_DIR / 'data' / 'churn_dataset.example.csv')
    assert len(frame) == 60
    assert set(frame.columns) == set(FEATURE_COLUMNS + ['churn'])
    assert frame['churn'].nunique() == 2


def test_final_feature_contract():
    assert len(NUMERIC_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 3
    assert set(FeatureVectorChurn.model_fields) == set(FEATURE_COLUMNS)


@pytest.mark.parametrize('kind,params', [
    ('logreg', {'C': 0.3, 'max_iter': 400}),
    ('random_forest', {'n_estimators': 7, 'max_depth': 3}),
])
def test_final_model_configuration(kind, params):
    clf = create_classifier(TrainingConfigChurn(model_type=kind, hyperparameters=params))
    assert isinstance(clf.get_params(), dict)
    for key, value in params.items():
        assert clf.get_params()[key] == value


def test_final_openapi_contains_all_routes(isolated_api):
    paths = isolated_api.get('/openapi.json').json()['paths']
    for path in ('/', '/health', '/dataset/preview', '/dataset/info',
                 '/dataset/split-info', '/model/train', '/model/status',
                 '/model/schema', '/model/metrics', '/predict'):
        assert path in paths


def test_final_retraining_keeps_saved_history(isolated_api, sample_client, isolated_paths):
    first = isolated_api.post('/model/train', json={'model_type': 'logreg', 'hyperparameters': {'max_iter': 200}})
    second = isolated_api.post('/model/train', json={'model_type': 'random_forest', 'hyperparameters': {'n_estimators': 8}})
    assert first.status_code == second.status_code == 200
    assert isolated_api.get('/model/metrics').json()['total'] == 2
    assert isolated_paths['history'].is_file()
    assert isolated_paths['model'].is_file()
    assert isolated_api.post('/predict', json=sample_client).status_code == 200


def test_final_health_and_docs_accessible_without_training(isolated_api):
    assert isolated_api.get('/health').status_code == 200
    assert isolated_api.get('/docs').status_code == 200
    assert isolated_api.get('/model/status').json()['trained'] is False
