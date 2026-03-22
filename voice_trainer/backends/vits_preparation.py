from __future__ import annotations

import json
from pathlib import Path

from ..vits import text as vits_text

_TAGGED_MULTILINGUAL_CLEANERS = {
    "cjks_cleaners",
    "cjke_cleaners",
    "cjke_cleaners2",
    "zh_ja_mixture_cleaners",
    "chinese_dialect_cleaners",
}

_LANGUAGE_TAGS = {
    "ko": "KO",
    "kr": "KO",
    "korean": "KO",
    "en": "EN",
    "english": "EN",
    "ja": "JA",
    "jp": "JA",
    "japanese": "JA",
    "zh": "ZH",
    "cn": "ZH",
    "chinese": "ZH",
}

_DEFAULT_KOREAN_SYMBOLS = [
    "_", ",", ".", "!", "?", "…", "~", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ",
    "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅏ",
    "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ", " ",
]


def _write_filelist(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")


def _uses_tagged_multilingual_cleaners(cleaner_names: list[str]) -> bool:
    return any(name in _TAGGED_MULTILINGUAL_CLEANERS for name in cleaner_names)


def _adapt_text_for_cleaners(text: str, cleaner_names: list[str], language: str) -> str:
    if not _uses_tagged_multilingual_cleaners(cleaner_names):
        return text
    tag = _LANGUAGE_TAGS.get(language.strip().lower())
    if tag is None:
        raise ValueError(f"Unsupported language for tagged multilingual cleaners: {language}")
    return f"[{tag}]{text}[{tag}]"


def _validate_cleaned_text(original_text: str, cleaned_text: str, symbols: list[str]) -> None:
    symbol_set = set(symbols)
    matched_symbol_count = sum(1 for symbol in cleaned_text if symbol in symbol_set)
    if matched_symbol_count == 0:
        raise ValueError(f"Cleaner removed all symbols from text: {original_text}")
    min_expected = max(3, len(cleaned_text.replace(" ", "")) // 3)
    if matched_symbol_count < min_expected:
        raise ValueError(
            "Cleaner output is not compatible with the configured symbols: "
            f"{original_text} -> {cleaned_text}"
        )


def _write_cleaned_filelist(
    path: Path,
    cleaner_names: list[str],
    language: str,
    symbols: list[str],
    text_index: int = 2,
) -> Path:
    cleaned_path = Path(str(path) + ".cleaned")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            adapted_text = _adapt_text_for_cleaners(parts[text_index], cleaner_names, language)
            cleaned_text = vits_text._clean_text(adapted_text, cleaner_names)
            _validate_cleaned_text(parts[text_index], cleaned_text, symbols)
            parts[text_index] = cleaned_text
            rows.append("|".join(parts))
    _write_filelist(cleaned_path, rows)
    return cleaned_path


def load_pretrained_vits_config(pretrained_generator: str) -> dict | None:
    if not pretrained_generator:
        return None
    config_path = Path(pretrained_generator).resolve().parent / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    language: str,
) -> dict:
    pretrained_config = load_pretrained_vits_config(pretrained_generator)
    pretrained_data = pretrained_config.get("data", {}) if pretrained_config else {}
    pretrained_model = pretrained_config.get("model", {}) if pretrained_config else {}
    pretrained_symbols = pretrained_config.get("symbols") if pretrained_config else None
    pretrained_speakers = pretrained_config.get("speakers") if pretrained_config else None
    symbols = pretrained_symbols if pretrained_symbols else _DEFAULT_KOREAN_SYMBOLS
    text_cleaners = pretrained_data.get("text_cleaners", ["korean_cleaners"])
    n_speakers = max(1, int(pretrained_data.get("n_speakers", 1)))
    cleaned_train_filelist = _write_cleaned_filelist(Path(train_filelist), text_cleaners, language, symbols)
    cleaned_val_filelist = _write_cleaned_filelist(Path(val_filelist), text_cleaners, language, symbols)

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
            "pretrained_discriminator": pretrained_discriminator,
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
            "cleaned_text": True,
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
            "gin_channels": pretrained_model.get("gin_channels", 256),
        },
        "speakers": pretrained_speakers if pretrained_speakers else ["speaker0"],
        "symbols": symbols,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return config
