from contextlib import asynccontextmanager
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from fastapi import FastAPI

from dataset_service import load_dataset
from error_handlers import ServiceError, register_error_handlers
from model_storage import load_churn_model
from monitoring import configure_logging
from app.api.routes import create_router

logger = configure_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset_loaded = False
    try:
        df = load_dataset()
    except (FileNotFoundError, EmptyDataError, ValueError, ParserError, UnicodeDecodeError) as exc:
        logger.warning('event=dataset_startup_unavailable error_type=%s', type(exc).__name__)
    else:
        app.state.dataset_loaded = True
        logger.info('event=dataset_loaded rows=%d columns=%d', *df.shape)
    app.state.model_bundle = load_churn_model()
    logger.info('event=service_started dataset_loaded=%s model_available=%s',
                app.state.dataset_loaded, app.state.model_bundle is not None)
    try:
        yield
    finally:
        app.state.model_bundle = None
        app.state.dataset_loaded = False
        logger.info('event=service_stopped')

def get_dataset() -> pd.DataFrame:
    try:
        df = load_dataset()
        app.state.dataset_loaded = True
        logger.info('event=dataset_loaded rows=%d columns=%d', *df.shape)
        return df
    except FileNotFoundError as exc:
        app.state.dataset_loaded = False
        raise ServiceError('DATASET_NOT_FOUND', 'Файл churn_dataset.csv не найден', 404) from exc
    except EmptyDataError as exc:
        app.state.dataset_loaded = False
        raise ServiceError('DATASET_EMPTY', 'CSV-файл пуст', 422) from exc
    except (ValueError, ParserError, UnicodeDecodeError) as exc:
        app.state.dataset_loaded = False
        reason = str(exc)
        if 'пуст' in reason.lower() or 'не содержит строк' in reason.lower():
            raise ServiceError('DATASET_EMPTY', 'Датасет не содержит записей', 422) from exc
        raise ServiceError('DATASET_INVALID', 'Некорректный формат или содержимое датасета', 422) from exc

app = FastAPI(title='ML Churn Prediction API',
              description='Service for customer churn prediction',
              version='1.0.0', lifespan=lifespan)
register_error_handlers(app)

app.include_router(create_router(dataset_loader=lambda: get_dataset()))