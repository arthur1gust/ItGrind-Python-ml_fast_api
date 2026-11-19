"""Совместимый импорт: ошибки перенесены в app.core.errors."""
from app.core.errors import (ErrorResponse, ServiceError, error_json,
    register_error_handlers, TRAIN_ERROR_RESPONSES, PREDICT_ERROR_RESPONSES)
