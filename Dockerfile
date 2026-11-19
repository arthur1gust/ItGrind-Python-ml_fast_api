# Python 3.13.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Копируем зависимости отдельно, чтобы Docker кэшировал pip install.
COPY requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# Копируем только код из контекста; реальный CSV исключён из image через .dockerignore.
COPY . .

# models/ используется для joblib и JSON.
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
