"""Настройка логирования ключевых событий churn API."""

import logging
import os
import sys


def configure_logging() -> logging.Logger:
    """Один stdout-handler на процесс, в том числе при uvicorn --reload."""
    logger = logging.getLogger("churn_service")
    name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    logger.setLevel(level)

    if not any(getattr(h, "_churn_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler._churn_handler = True
        logger.addHandler(handler)

    # Исключаем дублирование при собственной конфигурации логов Uvicorn.
    logger.propagate = False
    return logger
