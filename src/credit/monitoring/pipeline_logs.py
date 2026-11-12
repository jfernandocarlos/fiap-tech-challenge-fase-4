"""Logs de execução do pipeline (sucesso/falha) em JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit.config import PIPELINE_LOG
from credit.logging_config import get_logger

logger = get_logger(__name__)


def append_event(
    event: str,
    status: str,
    details: dict[str, Any] | None = None,
    path: Path = PIPELINE_LOG,
) -> None:
    """Append de um evento operacional no arquivo JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "details": details or {},
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("pipeline_event", step=event, status=status, details=details or {})


def read_events(path: Path = PIPELINE_LOG) -> list[dict[str, Any]]:
    """Lê todos os eventos do log do pipeline."""
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def summarize_health(path: Path = PIPELINE_LOG) -> dict[str, Any]:
    """Resume saúde operacional a partir do JSONL."""
    events = read_events(path)
    ok = sum(1 for e in events if e.get("status") == "success")
    fail = sum(1 for e in events if e.get("status") == "failure")
    alerts = sum(1 for e in events if e.get("status") == "alert")
    return {
        "total_events": len(events),
        "success": ok,
        "failure": fail,
        "alerts": alerts,
        "last_event": events[-1] if events else None,
    }
