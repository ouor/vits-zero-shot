from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .audio import write_json
from .mini_vits import MiniVITS
from .text import TextTokenizer
from .vits_data import TTSDataset, collate_tts_batch


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_dataloader(config: dict, filelist_path: str, shuffle: bool) -> DataLoader:
    tokenizer = TextTokenizer(config["symbols"])
    data = config["data"]
    dataset = TTSDataset(
        filelist_path=filelist_path,
        tokenizer=tokenizer,
        sample_rate=data["sampling_rate"],
        n_fft=data["filter_length"],
        hop_length=data["hop_length"],
        win_length=data["win_length"],
        n_mels=data["n_mel_channels"],
        mel_fmin=data["mel_fmin"],
        mel_fmax=data["mel_fmax"],
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_tts_batch,
    )


def train_vits(config_path: str | Path, output_dir: str | Path) -> dict:
    config = _load_config(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader = _build_dataloader(config, config["data"]["training_files"], shuffle=True)
    val_loader = _build_dataloader(config, config["data"]["validation_files"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniVITS(
        vocab_size=len(config["symbols"]),
        mel_dim=config["data"]["n_mel_channels"],
        hidden_dim=config["model"]["hidden_channels"],
        latent_dim=config["model"]["inter_channels"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=tuple(config["train"]["betas"]),
        eps=config["train"]["eps"],
    )

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                tokens=batch["tokens"],
                token_lengths=batch["token_lengths"],
                mels=batch["mels"],
                mel_lengths=batch["mel_lengths"],
            )
            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(outputs["loss"].item())

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    tokens=batch["tokens"],
                    token_lengths=batch["token_lengths"],
                    mels=batch["mels"],
                    mel_lengths=batch["mel_lengths"],
                )
                val_total += float(outputs["loss"].item())

        train_loss = train_total / max(len(train_loader), 1)
        val_loss = val_total / max(len(val_loader), 1)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(epoch_summary)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        torch.save(checkpoint, output_root / "last.ckpt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_root / "best.ckpt")

    summary = {
        "device": str(device),
        "epochs": config["train"]["epochs"],
        "best_val_loss": best_val_loss,
        "history": history,
    }
    write_json(output_root / "training_summary.json", summary)
    return summary
