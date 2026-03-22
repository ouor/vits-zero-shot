from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    ref_root: Path
    faster_qwen_root: Path
    speechbrain_root: Path
    vits_root: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_paths() -> PipelinePaths:
    root = repo_root()
    ref_root = root / ".ref"
    return PipelinePaths(
        root=root,
        ref_root=ref_root,
        faster_qwen_root=ref_root / "faster-qwen3-tts",
        speechbrain_root=ref_root / "speechbrain",
        vits_root=ref_root / "vits",
    )


def resolve_from_root(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate
