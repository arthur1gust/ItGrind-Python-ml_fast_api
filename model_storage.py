"""Совместимый интерфейс Day 6. Путь изолируется через monkeypatch."""
from app.core.config import MODEL_PATH as _DEFAULT_PATH
from app.ml.storage import load_bundle, save_bundle
MODEL_PATH = _DEFAULT_PATH

def save_churn_model(model, metrics, train_size, test_size):
    return save_bundle(model, metrics, train_size, test_size, MODEL_PATH)

def load_churn_model():
    return load_bundle(MODEL_PATH)
