# ML FastAPI — Customer Churn Prediction

Учебный REST API для **прогнозирования оттока клиентов** (`churn`: `1` — вероятный уход, `0` — вероятное сохранение). Проект объединяет подготовку данных в `scikit-learn Pipeline`, обучение двух классификаторов, предсказания через FastAPI, историю метрик, тестирование и контейнеризацию.

Реализовано в рамках Days 1–14. Код рассчитан на Python **3.13** и Pydantic **v2**.

## Функциональность

- Загрузка и проверка CSV через pandas; просмотр данных и распределения классов.
- Одинаковый контракт признаков при обучении и `/predict`.
- `ColumnTransformer`: числовые поля → `SimpleImputer(median)` → `StandardScaler`; категориальные → `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`.
- `LogisticRegression` и `RandomForestClassifier` с выбором параметров через API.
- Accuracy, F1, ROC-AUC (когда оба класса присутствуют в тестовой выборке).
- Сохранение полного `Pipeline` в `models/churn_model.joblib` и журнала экспериментов в `models/training_history.json`.
- Единые ответы об ошибках (`code`, `message`, `details`), события логирования, `/health`.
- `pytest`, `TestClient`, синтетические fixture и Docker.

## Структура проекта

```text
ml_fast_api/
├── main.py                        # ASGI-приложение, lifespan, подключения
├── app/
│   ├── api/routes.py              # HTTP-эндпоинты
│   ├── core/
│   │   ├── config.py              # Пути к CSV, joblib и JSON
│   │   ├── dataset.py             # Чтение/проверка CSV
│   │   ├── errors.py              # Глобальные обработчики ошибок
│   │   └── logging.py             # Логирование событий
│   ├── ml/
│   │   ├── preprocessing.py       # Набор признаков и ColumnTransformer
│   │   ├── service.py             # Обучение, классификаторы, метрики
│   │   ├── storage.py             # Сохранение/загрузка модели
│   │   └── history.py             # История обучений
│   └── schemas/churn.py           # Pydantic-схемы
├── data/churn_dataset.example.csv # Синтетические демонстрационные данные
├── tests/                         # Unit + integration тесты Days 7–14
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── requirements-docker.txt
```

Файлы `schemas.py`, `preprocessing.py`, `ml_service.py`, `model_storage.py`, `dataset_service.py`, `training_history.py`, `monitoring.py`, `error_handlers.py` в корне **сохранены как совместимые адаптеры**, чтобы старые импорты и pytest-тесты Days 7–13 продолжали работать. Новую логику следует добавлять в соответствующий пакет `app/`.

## Формат churn_dataset.csv

В проекте используется **10 столбцов**: 9 признаков и бинарная целевая переменная `churn`.

| Поле | Тип | Значение |
|---|---|---|
| `monthly_fee` | float | Месячная оплата |
| `usage_hours` | float | Использование продукта, часы |
| `support_requests` | int | Количество обращений в поддержку |
| `account_age_months` | int | Стаж клиента, месяцы |
| `failed_payments` | int | Неудачные платежи |
| `region` | str | Регион (категория) |
| `device_type` | str | Тип устройства (категория) |
| `payment_method` | str | Способ оплаты (категория) |
| `autopay_enabled` | int | Автоплатёж: 0 или 1 |
| `churn` | int | Цель: 0 или 1; **не передавать в `/predict`** |

Пример строки:

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
49.99,12.5,3,18,1,europe,mobile,card,1,1
```

CSV должен иметь заголовки, все указанные столбцы и непустые строки. Категории кодируются автоматически. Для обучения и оценки нужно достаточно примеров **обоих классов**. Распределение классов учитывается при `train_test_split(stratify=y, random_state=42)`.

**Важно:** файл `data/churn_dataset.csv` намеренно **не включён в архив и Git**, чтобы не перезаписывать твой рабочий датасет. В архиве есть `data/churn_dataset.example.csv` с 60 синтетическими клиентами для демонстрации. Если у тебя ещё нет CSV, создай рабочую копию:

```powershell
Copy-Item data/churn_dataset.example.csv data/churn_dataset.csv
```

Если реальный CSV уже существует, **не выполняй копирование** — используй имеющийся файл.

## Локальный запуск (Windows PowerShell / PyCharm)

Из папки `C:\project\ItGrind\python\ml_fast_api`:

```powershell
# Если окружение уже создано, пропусти первые две команды
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements-dev.txt
python -m uvicorn main:app --reload
```

Открой Swagger: <http://127.0.0.1:8000/docs>. Сервис: <http://127.0.0.1:8000/>. `/health` — <http://127.0.0.1:8000/health>.

Запись модели/истории производится в `models/`. Удалять их перед обновлением исходного кода **не нужно**. Файлы `.joblib` нужно загружать только из доверенных источников.

## API

| HTTP | Путь | Назначение |
|---|---|---|
| GET | `/` | Проверка ответа API |
| GET | `/health` | Загружен ли датасет и доступна ли модель |
| GET | `/dataset/preview?n=5` | Первые N строк CSV |
| GET | `/dataset/info` | Размер, колонки, распределение классов |
| GET | `/dataset/split-info` | Train/test и параметры предобработки |
| POST | `/model/train` | Обучить модель и сохранить результат |
| GET | `/model/status` | Последняя модель и её метрики |
| GET | `/model/schema` | Девять признаков, их категории и типы |
| GET | `/model/metrics?limit=5` | История обучений и фильтр `model_type` |
| POST | `/predict` | Прогноз для одного клиента или списка |

### POST /model/train

В Swagger выбери `POST /model/train` → **Try it out** и отправь JSON для логистической регрессии:

```json
{
  "model_type": "logreg",
  "hyperparameters": {"C": 0.5, "max_iter": 1000}
}
```

Или для случайного леса:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {"n_estimators": 150, "max_depth": 8}
}
```

Успешный ответ содержит `model_type`, `trained_at`, `accuracy`, `f1`, `roc_auc`, размер train/test и фактические гиперпараметры. `GET /model/metrics?model_type=logreg` позволит сравнить только обучения LogisticRegression.

### POST /predict

После обучения отправь **ровно девять признаков без `churn`**:

```json
{
  "monthly_fee": 49.99,
  "usage_hours": 12.5,
  "support_requests": 3,
  "account_age_months": 18,
  "failed_payments": 1,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card",
  "autopay_enabled": 1
}
```

Пример **формата** ответа (числа иллюстративные, зависят от модели):

```json
{"churn": 1, "probabilities": {"0": 0.21, "1": 0.79}}
```

`/predict` принимает также массив JSON-объектов клиентов и возвращает массив предсказаний. Порядок ключей в JSON может быть любым. Увидеть ожидаемые типы можно в `GET /model/schema`.

### Ошибки и диагностика

Для ошибок используется единый формат:

```json
{
  "code": "MODEL_NOT_TRAINED",
  "message": "Модель не обучена. Сначала выполните POST /model/train",
  "details": []
}
```

Основные HTTP-статусы: `422` (неверные признаки, конфигурация, пустой CSV), `404` (CSV не найден), `503` (модель не обучена), `500` (непредвиденная ошибка). Техническая трассировка пишется в серверный лог и не отдаётся пользователю.

Пример `/health` до обучения модели при доступном CSV:

```json
{"status": "ok", "model_available": false, "dataset_loaded": true}
```

`/health` возвращает HTTP 200, если веб-процесс отвечает, даже при статусе `degraded`. Это **диагностический endpoint**, а не строгая проверка готовности модели для `/predict`.

## Docker Desktop / Windows

Запусти Docker Desktop в режиме Linux containers. **Поскольку CSV не включён в образ, папка `data/` подключается извне (в исправленном Day 13 это уже учтено).** Это важно: `.dockerignore` исключает рабочий `churn_dataset.csv` из образа, чтобы он не попадал в Docker image или Git.

**Рекомендуемый запуск через Compose (конфигурация из исправленного Day 13):**

```powershell
Test-Path .\data\churn_dataset.csv  # должно быть True
docker compose up --build -d
```

**Альтернатива — Docker вручную:**

```powershell
# Выполняй из корня проекта; сначала убедись, что data/churn_dataset.csv существует
docker version
docker build -t ml-fastapi-churn:final .
docker run -d --name churn-final -p 8000:8000 -v "${PWD}/data:/app/data:ro" -v churn-models:/app/models ml-fastapi-churn:final
```

Если порт 8000 занят локальным Uvicorn, предварительно останови локальный сервер (Ctrl+C). После запуска:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/docs -UseBasicParsing | Select-Object StatusCode

docker exec churn-final python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"
docker exec churn-final python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/docs').status)"
docker logs --tail 50 churn-final
docker inspect --format '{{.State.Health.Status}}' churn-final
```

Ожидаемые HTTP-статусы `/health` и `/docs` — `200`. Модель в новом контейнере отсутствует до первого `/model/train`, после этого появится в общем Docker volume `churn-models` и сохранится при пересоздании контейнера.

**Остановка:** `docker stop churn-final`, затем `docker rm churn-final` (не удаляет named volume).

## Тестирование

```powershell
python -m pytest tests/test_day14_final.py -v
python -m pytest -v
```

Тесты работают на синтетическом датасете (60 клиентов) с изоляцией CSV, `.joblib` и истории через `pytest` fixtures / `tmp_path` / `monkeypatch`; запуск Uvicorn для pytest не требуется.

## Переменные окружения

При необходимости можно задать `CHURN_DATASET_PATH`, `CHURN_MODEL_PATH`, `CHURN_HISTORY_PATH`, `LOG_LEVEL` (по умолчанию `INFO`) до запуска приложения. Не сохраняй `.env` с секретами в Git.

## Git / финальный коммит

```powershell
python -m pytest -v
git status
git add -A
git status
git commit -m "Day 14: refactor churn API, document setup and Docker"
git push origin main
```

Перед `git add -A` убедись, что `.gitignore` исключает `.venv/`, `data/churn_dataset.csv`, `models/*.joblib`, `models/training_history.json` и секреты. Не удаляй рабочие файлы после рефакторинга до успешного запуска и тестов.

## Ограничения

Это **учебный сервис**, не полноценное промышленное MLOps-решение: нет аутентификации, базы пользователей, CI/CD деплоя, модели реестра/версий, управления drift и эксплуатационных гарантий SLA. В одной работающей копии хранится одна актуальная модель; JSON сохраняет историю конфигураций и метрик, а не предыдущие бинарные модели. Детерминированность в пределах одинаковых версий библиотек достигается фиксированными `random_state`; при обновлении библиотек численные результаты могут немного изменяться.
