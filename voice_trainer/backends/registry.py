from __future__ import annotations

from .base import TrainingBackend
from .vits_backend import VitsBackend


def get_backend(name: str) -> TrainingBackend:
    normalized = name.strip().lower()
    if normalized == "vits":
        return VitsBackend()
    raise ValueError(f"Unsupported training backend: {name}")
