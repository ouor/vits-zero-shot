from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import load_waveform
from .features import MelExtractor
from .text import TextTokenizer


class TTSDataset(Dataset):
    def __init__(
        self,
        *,
        filelist_path: str | Path,
        tokenizer: TextTokenizer,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        mel_fmin: float,
        mel_fmax: float | None,
    ) -> None:
        self.items = self._load_items(filelist_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.mel_extractor = MelExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )

    @staticmethod
    def _load_items(filelist_path: str | Path) -> list[tuple[str, str]]:
        items = []
        with Path(filelist_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                wav_path, text = line.rstrip("\n").split("|", 1)
                items.append((wav_path, text))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        wav_path, text = self.items[index]
        waveform, _ = load_waveform(wav_path, sample_rate=self.sample_rate)
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        mel = self.mel_extractor(waveform)
        return {
            "tokens": tokens,
            "mel": mel.transpose(0, 1),
        }


def collate_tts_batch(batch: list[dict]) -> dict:
    token_lengths = torch.tensor([item["tokens"].size(0) for item in batch], dtype=torch.long)
    mel_lengths = torch.tensor([item["mel"].size(0) for item in batch], dtype=torch.long)

    max_tokens = int(token_lengths.max().item())
    max_mels = int(mel_lengths.max().item())
    mel_dim = batch[0]["mel"].size(1)

    tokens = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    mels = torch.zeros(len(batch), max_mels, mel_dim, dtype=torch.float32)

    for index, item in enumerate(batch):
        token_count = item["tokens"].size(0)
        mel_count = item["mel"].size(0)
        tokens[index, :token_count] = item["tokens"]
        mels[index, :mel_count] = item["mel"]

    return {
        "tokens": tokens,
        "token_lengths": token_lengths,
        "mels": mels,
        "mel_lengths": mel_lengths,
    }
