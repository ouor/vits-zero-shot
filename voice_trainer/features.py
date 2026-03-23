from __future__ import annotations

import torch
import torchaudio


class MelExtractor:
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        mel_fmin: float,
        mel_fmax: float | None,
    ) -> None:
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax,
            power=1.0,
            normalized=False,
            center=True,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)
