from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_text(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_waveform(path: str | Path, sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate is not None and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    return waveform, sr


def save_waveform(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), waveform.squeeze(0).cpu().numpy(), sample_rate, subtype="PCM_16")


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
