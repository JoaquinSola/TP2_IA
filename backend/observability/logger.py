import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_file = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _write(event: dict) -> None:
    with open(_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_llm_call(
    session_id: str,
    node: str,
    prompt_summary: str,
    response_summary: str,
    latency_ms: float,
    tokens_used: Optional[int] = None,
) -> None:
    _write({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "llm_call",
        "session_id": session_id,
        "node": node,
        "prompt_summary": prompt_summary,
        "response_summary": response_summary,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
    })


def log_tool_call(session_id: str, tool_name: str, inputs: Any) -> None:
    _write({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "tool_call",
        "session_id": session_id,
        "tool": tool_name,
        "inputs": _safe_serialize(inputs),
    })


def log_tool_result(session_id: str, tool_name: str, result: Any, latency_ms: float) -> None:
    _write({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "tool_result",
        "session_id": session_id,
        "tool": tool_name,
        "result": _safe_serialize(result),
        "latency_ms": round(latency_ms, 2),
    })


def log_error(session_id: str, node: str, error: str) -> None:
    _write({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "error",
        "session_id": session_id,
        "node": node,
        "error": error,
    })


def log_session_start(session_id: str) -> None:
    _write({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "session_start",
        "session_id": session_id,
    })


def get_session_logs(session_id: str) -> list[dict]:
    """Retorna todos los logs de una sesión (para el endpoint de observabilidad)."""
    logs = []
    if _log_file.exists():
        with open(_log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        logs.append(entry)
                except json.JSONDecodeError:
                    pass
    return logs


def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    return obj


class Timer:
    """Context manager para medir latencia."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
