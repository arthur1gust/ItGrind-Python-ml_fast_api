"""Расположение данных и результатов относительно корня репозитория."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(os.getenv('CHURN_DATASET_PATH', str(BASE_DIR / 'data' / 'churn_dataset.csv')))
MODEL_PATH = Path(os.getenv('CHURN_MODEL_PATH', str(BASE_DIR / 'models' / 'churn_model.joblib')))
HISTORY_PATH = Path(os.getenv('CHURN_HISTORY_PATH', str(BASE_DIR / 'models' / 'training_history.json')))
