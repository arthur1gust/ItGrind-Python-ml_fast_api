"""Фабрика классификаторов, обучение и оценка качества."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from app.ml.preprocessing import build_preprocessor
from app.schemas.churn import TrainingConfigChurn

def create_classifier(config: TrainingConfigChurn):
    params = dict(config.hyperparameters)
    if config.model_type == 'logreg':
        if 'C' in params and (not isinstance(params['C'], (int, float)) or isinstance(params['C'], bool) or params['C'] <= 0):
            raise ValueError('C должен быть положительным числом')
        if 'max_iter' in params and (type(params['max_iter']) is not int or params['max_iter'] < 1):
            raise ValueError('max_iter должен быть положительным целым числом')
        defaults = {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if config.model_type == 'random_forest':
        n = params.get('n_estimators', 100)
        if type(n) is not int or not (1 <= n <= 500):
            raise ValueError('n_estimators должен быть целым числом от 1 до 500')
        depth = params.get('max_depth')
        if depth is not None and (type(depth) is not int or depth <= 0):
            raise ValueError('max_depth должен быть положительным целым числом или null')
        defaults = {'n_estimators': 100, 'random_state': 42}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    raise ValueError(f'Неподдерживаемая модель: {config.model_type}')

def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series,
                      config: TrainingConfigChurn | None = None) -> Pipeline:
    config = config or TrainingConfigChurn()
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', create_classifier(config)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_churn_model(model: Pipeline, X_test: pd.DataFrame,
                         y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        'accuracy': round(float(accuracy_score(y_test, predictions)), 4),
        'f1': round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }
