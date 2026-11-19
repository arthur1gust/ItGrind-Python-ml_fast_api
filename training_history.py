"""Совместимый интерфейс Day 11. Путь можно подменять в pytest."""
from app.core.config import HISTORY_PATH as _DEFAULT_PATH
from app.ml.history import (read_training_history as _read, append_training_record as _append,
                            get_training_metrics as _metrics)
HISTORY_PATH = _DEFAULT_PATH

def read_training_history():
    return _read(HISTORY_PATH)

def append_training_record(model_bundle):
    return _append(model_bundle, HISTORY_PATH)

def get_training_metrics(limit=5, model_type=None):
    return _metrics(limit, model_type, path=HISTORY_PATH)
