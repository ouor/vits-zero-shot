from __future__ import annotations

import json
import random
from pathlib import Path

from .audio import load_waveform, save_waveform, write_json
from .full_vits import text as vits_text


def _write_filelist(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")


def _write_cleaned_filelist(path: Path, cleaner_names: list[str], text_index: int = 2) -> Path:
    cleaned_path = Path(str(path) + ".cleaned")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            parts[text_index] = vits_text._clean_text(parts[text_index], cleaner_names)
            rows.append("|".join(parts))
    _write_filelist(cleaned_path, rows)
    return cleaned_path


def _load_pretrained_config(pretrained_generator: str) -> dict | None:
    if not pretrained_generator:
        return None
    config_path = Path(pretrained_generator).resolve().parent / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export_vits_dataset(
    *,
    selected_candidates: list[dict],
    output_dir: Path,
    target_sample_rate: int,
    train_split_ratio: float,
    random_seed: int = 1234,
) -> dict:
    if len(selected_candidates) < 2:
        raise ValueError("VITS dataset export requires at least 2 selected candidates.")
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
        rows.append(f"{wav_path}|0|{candidate['text']}")
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
    pretrained_generator: str = "",
    pretrained_discriminator: str = "",
) -> dict:
    pretrained_config = _load_pretrained_config(pretrained_generator)
    pretrained_data = pretrained_config.get("data", {}) if pretrained_config else {}
    pretrained_model = pretrained_config.get("model", {}) if pretrained_config else {}
    pretrained_symbols = pretrained_config.get("symbols") if pretrained_config else None
    pretrained_speakers = pretrained_config.get("speakers") if pretrained_config else None
    text_cleaners = pretrained_data.get("text_cleaners", ["korean_cleaners"])
    n_speakers = max(1, int(pretrained_data.get("n_speakers", 1)))
    cleaned_train_filelist = _write_cleaned_filelist(Path(train_filelist), text_cleaners)
    cleaned_val_filelist = _write_cleaned_filelist(Path(val_filelist), text_cleaners)

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
            "num_gpus": 1,
            "c_mel": 45,
            "c_kl": 1.0,
            "pretrained_generator": pretrained_generator,
            "pretrained_discriminator": pretrained_discriminator
        },
        "data": {
            "training_files": str(cleaned_train_filelist),
            "validation_files": str(cleaned_val_filelist),
            "text_cleaners": text_cleaners,
            "max_wav_value": 32768.0,
            "sampling_rate": sampling_rate,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": pretrained_data.get("add_blank", True),
            "n_speakers": n_speakers,
            "cleaned_text": True
        },
        "model": {
            "inter_channels": pretrained_model.get("inter_channels", 128),
            "hidden_channels": pretrained_model.get("hidden_channels", 192),
            "filter_channels": pretrained_model.get("filter_channels", 768),
            "n_heads": pretrained_model.get("n_heads", 2),
            "n_layers": pretrained_model.get("n_layers", 6),
            "kernel_size": pretrained_model.get("kernel_size", 3),
            "p_dropout": pretrained_model.get("p_dropout", 0.1),
            "resblock": pretrained_model.get("resblock", "1"),
            "resblock_kernel_sizes": pretrained_model.get("resblock_kernel_sizes", [3, 7, 11]),
            "resblock_dilation_sizes": pretrained_model.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            "upsample_rates": pretrained_model.get("upsample_rates", [8, 8, 2, 2]),
            "upsample_initial_channel": pretrained_model.get("upsample_initial_channel", 512),
            "upsample_kernel_sizes": pretrained_model.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            "n_layers_q": pretrained_model.get("n_layers_q", 3),
            "use_spectral_norm": pretrained_model.get("use_spectral_norm", False),
            "gin_channels": pretrained_model.get("gin_channels", 256)
        },
        "speakers": pretrained_speakers if pretrained_speakers else ["speaker0"],
        "symbols": pretrained_symbols if pretrained_symbols else [
            "_", ",", ".", "!", "?", "…", "~", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ",
            "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅏ",
            "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ", " "
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return config
