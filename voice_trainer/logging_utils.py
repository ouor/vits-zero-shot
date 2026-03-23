from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def get_run_logger(run_root: Path) -> logging.Logger:
    logger = logging.getLogger(f"voice_trainer.run.{run_root.name}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    run_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_root / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def append_event(run_root: Path, payload: dict) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    event = _json_safe({"ts": _utc_timestamp(), **payload})
    with (run_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _format_fields(fields: dict) -> str:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return " ".join(parts)


def log_stage_start(
    logger: logging.Logger,
    run_root: Path,
    *,
    stage: str,
    index: int,
    total: int,
    **fields,
) -> None:
    message = f"[{index}/{total}] {stage} started"
    field_suffix = _format_fields(fields)
    if field_suffix:
        message = f"{message} | {field_suffix}"
    logger.info(message)
    append_event(
        run_root,
        {
            "stage": stage,
            "status": "started",
            "index": index,
            "total": total,
            "fields": fields,
        },
    )


def log_stage_end(
    logger: logging.Logger,
    run_root: Path,
    *,
    stage: str,
    index: int,
    total: int,
    elapsed_sec: float,
    **fields,
) -> None:
    message = f"[{index}/{total}] {stage} completed in {elapsed_sec:.1f}s"
    field_suffix = _format_fields(fields)
    if field_suffix:
        message = f"{message} | {field_suffix}"
    logger.info(message)
    append_event(
        run_root,
        {
            "stage": stage,
            "status": "completed",
            "index": index,
            "total": total,
            "elapsed_sec": round(elapsed_sec, 3),
            "fields": fields,
        },
    )


def log_stage_error(
    logger: logging.Logger,
    run_root: Path,
    *,
    stage: str,
    index: int,
    total: int,
    elapsed_sec: float,
    error: Exception,
) -> None:
    logger.exception(
        "[%s/%s] %s failed after %.1fs | error=%s",
        index,
        total,
        stage,
        elapsed_sec,
        error,
    )
    append_event(
        run_root,
        {
            "stage": stage,
            "status": "failed",
            "index": index,
            "total": total,
            "elapsed_sec": round(elapsed_sec, 3),
            "error": str(error),
        },
    )


@contextmanager
def stage_timer() -> Iterator[callable]:
    started_at = time.perf_counter()
    yield lambda: time.perf_counter() - started_at
