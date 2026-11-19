"""Единый формат ошибок и глобальные обработчики FastAPI."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("churn_service.errors")


class ErrorResponse(BaseModel):
    code: str = Field(description="Машинно-читаемый код ошибки")
    message: str = Field(description="Понятное описание ошибки")
    details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Сведения о некорректных полях или причинах ошибки",
    )


class ServiceError(Exception):
    """Ожидаемая ошибка бизнес-логики с безопасным сообщением для клиента."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 422,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []


def error_json(
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=code, message=message, details=details or []
        ).model_dump(mode="json"),
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Подключаем обработчики к конкретному приложению FastAPI."""

    @app.exception_handler(ServiceError)
    async def service_exception_handler(request: Request, exc: ServiceError):
        logger.warning(
            "event=service_error path=%s code=%s status=%d",
            request.url.path, exc.code, exc.status_code,
        )
        return error_json(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "event=request_validation_failed path=%s error_count=%d",
            request.url.path, len(exc.errors()),
        )
        # Не возвращаем exc.body/input или repr(exc): они могут содержать личные данные.
        details = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())),
                "reason": str(err.get("msg", "Некорректное значение")),
                "type": str(err.get("type", "invalid")),
            }
            for err in exc.errors()
        ]
        return error_json(
            422, "VALIDATION_ERROR", "Некорректные входные данные", details
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(
            "event=http_error path=%s status=%d",
            request.url.path, exc.status_code,
        )
        # Стандартные HTTP-ошибки FastAPI/Starlette, включая неизвестный URL.
        if exc.status_code in (204, 304):
            return Response(status_code=exc.status_code, headers=exc.headers)
        codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 503: "SERVICE_UNAVAILABLE"}
        message = exc.detail if isinstance(exc.detail, str) else "Ошибка HTTP-запроса"
        return error_json(
            exc.status_code,
            codes.get(exc.status_code, "HTTP_ERROR"),
            message,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        # Трассировка остаётся в логах сервера, но не передаётся клиенту.
        logger.exception("event=unexpected_error path=%s", request.url.path)
        return error_json(500, "INTERNAL_ERROR", "Внутренняя ошибка сервера")


TRAIN_ERROR_RESPONSES = {
    404: {
        "model": ErrorResponse,
        "description": "Файл датасета отсутствует",
        "content": {"application/json": {"example": {
            "code": "DATASET_NOT_FOUND",
            "message": "Файл churn_dataset.csv не найден",
            "details": [],
        }}},
    },
    422: {
        "model": ErrorResponse,
        "description": "Неверная конфигурация или некорректный датасет",
        "content": {"application/json": {"example": {
            "code": "INVALID_TRAINING_CONFIG",
            "message": "Некорректные параметры обучения",
            "details": [{"reason": "n_estimators должен быть целым числом от 1 до 500"}],
        }}},
    },
    500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервиса"},
}

PREDICT_ERROR_RESPONSES = {
    422: {
        "model": ErrorResponse,
        "description": "Неверный набор или типы признаков",
        "content": {"application/json": {"example": {
            "code": "VALIDATION_ERROR",
            "message": "Некорректные входные данные",
            "details": [{
                "field": "body.monthly_fee",
                "reason": "Input should be a valid number",
                "type": "float_parsing",
            }],
        }}},
    },
    503: {
        "model": ErrorResponse,
        "description": "Обученная модель отсутствует",
        "content": {"application/json": {"example": {
            "code": "MODEL_NOT_TRAINED",
            "message": "Модель не обучена. Сначала выполните POST /model/train",
            "details": [],
        }}},
    },
    500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервиса"},
}
