from __future__ import annotations

import random
from pathlib import Path

from ..audio import load_waveform, save_waveform, write_json


def _write_filelist(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")


def export_selected_corpus(
    *,
    selected_candidates: list[dict],
    output_dir: Path,
    target_sample_rate: int,
    train_split_ratio: float,
    language: str,
    random_seed: int = 1234,
) -> dict:
    if len(selected_candidates) < 2:
        raise ValueError("Corpus export requires at least 2 selected candidates.")
    if not 0.0 < train_split_ratio < 1.0:
        raise ValueError("train_split_ratio must be between 0 and 1.")

    dataset_dir = output_dir / "dataset"
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest = []
    for index, candidate in enumerate(selected_candidates, start=1):
        waveform, _ = load_waveform(candidate["wav_path"], sample_rate=target_sample_rate)
        filename = f"sample_{index:04d}.wav"
        wav_path = wav_dir / filename
        save_waveform(wav_path, waveform, target_sample_rate)
        candidate_language = candidate.get("language", language)
        rows.append(f"{wav_path}|0|{candidate['text']}")
        manifest.append(
            {
                "id": candidate["id"],
                "text": candidate["text"],
                "language": candidate_language,
                "speaker_similarity": candidate["speaker_similarity"],
                "wav_path": str(wav_path),
                "speaker_id": 0,
            }
        )

    rng = random.Random(random_seed)
    rng.shuffle(rows)

    split_index = max(1, int(len(rows) * train_split_ratio))
    if split_index >= len(rows):
        split_index = len(rows) - 1

    train_rows = rows[:split_index]
    val_rows = rows[split_index:]

    filelists_dir = output_dir / "filelists"
    train_filelist = filelists_dir / "train.txt"
    val_filelist = filelists_dir / "val.txt"
    _write_filelist(train_filelist, train_rows)
    _write_filelist(val_filelist, val_rows)

    manifest_path = output_dir / "selected_dataset.json"
    write_json(manifest_path, {"items": manifest})

    return {
        "dataset_dir": str(dataset_dir),
        "train_filelist": str(train_filelist),
        "val_filelist": str(val_filelist),
        "manifest_path": str(manifest_path),
        "item_count": len(manifest),
        "language": language,
    }
