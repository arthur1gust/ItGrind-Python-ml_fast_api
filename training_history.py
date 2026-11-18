import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any


HISTORY_PATH = (
    Path(__file__).resolve().parent / "models" / "training_history.json"
)

# Защищаем read -> append -> replace от одновременных вызовов в одном процессе.
_history_lock = Lock()


def read_training_history() -> list[dict[str, Any]]:
    """Читаем всю историю в порядке добавления (старые -> новые)."""
    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("Некорректный формат файла истории обучений")

    return records


def append_training_record(model_bundle: dict[str, Any]) -> dict[str, Any]:
    """Записываем успешное обучение, не теряя предыдущие записи."""
    record = {
        "timestamp": model_bundle["trained_at"],
        "model_type": model_bundle["model_type"],
        "hyperparameters": model_bundle["hyperparameters"],
        "metrics": model_bundle["metrics"],
        "train_size": model_bundle["train_size"],
        "test_size": model_bundle["test_size"],
    }

    with _history_lock:
        records = read_training_history()
        records.append(record)
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Пишем в файл рядом с целевым и атомарно заменяем исходный JSON.
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=HISTORY_PATH.parent,
                prefix=".training_history_",
                suffix=".tmp",
                delete=False,
            ) as temp:
                temporary_path = Path(temp.name)
                json.dump(records, temp, ensure_ascii=False, indent=2, allow_nan=False)
                temp.write("\n")
                temp.flush()
                os.fsync(temp.fileno())

            os.replace(temporary_path, HISTORY_PATH)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return record


def get_training_metrics(
    limit: int = 5,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Последнее обучение и последние N записей, опционально по алгоритму."""
    records = read_training_history()

    if model_type is not None:
        records = [
            record for record in records
            if record.get("model_type") == model_type
        ]

    # История хранится старые -> новые. API отдаёт новые -> старые.
    latest_first = list(reversed(records))
    latest = latest_first[0] if latest_first else None

    return {
        "latest": latest,
        "total": len(records),
        "history": latest_first[:limit],
    }
