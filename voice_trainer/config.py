from __future__ import annotations

import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_from_root(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def get_training_backend_name(config: dict) -> str:
    return config.get("training_backend", "vits")


def get_backend_config(config: dict, backend_name: str) -> dict:
    backends = config.get("backends", {})
    if backend_name in backends:
        return backends[backend_name]

    legacy_backend_config = config.get(backend_name)
    if legacy_backend_config is not None:
        return legacy_backend_config

    raise KeyError(f"Missing configuration for training backend: {backend_name}")
