from app.core.config import DATASET_PATH as _DEFAULT_PATH
from app.core.dataset import read_churn_csv
DATASET_PATH = _DEFAULT_PATH

def load_dataset():
    return read_churn_csv(DATASET_PATH)
