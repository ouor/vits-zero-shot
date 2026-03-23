from __future__ import annotations

import json
import random
from pathlib import Path

from .audio import load_waveform, save_waveform, write_json


def _write_filelist(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")


def export_vits_dataset(
    *,
    selected_candidates: list[dict],
    output_dir: Path,
    target_sample_rate: int,
    train_split_ratio: float,
    random_seed: int = 1234,
) -> dict:
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
        rows.append(f"{wav_path}|{candidate['text']}")
        manifest.append(
            {
                "id": candidate["id"],
                "text": candidate["text"],
                "speaker_similarity": candidate["speaker_similarity"],
                "wav_path": str(wav_path),
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
    }


def build_vits_config(
    *,
    output_path: Path,
    train_filelist: str,
    val_filelist: str,
    batch_size: int,
    epochs: int,
    sampling_rate: int,
) -> dict:
    config = {
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": epochs,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": batch_size,
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0
        },
        "data": {
            "training_files": train_filelist,
            "validation_files": val_filelist,
            "text_cleaners": ["korean_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": sampling_rate,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": False
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 256
        },
        "symbols": [
            "_", ",", ".", "!", "?", "…", "~", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ",
            "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅏ",
            "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ", " "
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return config
