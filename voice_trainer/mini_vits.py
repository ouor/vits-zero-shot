from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def generate_path(token_lengths: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
    batch_size = token_lengths.size(0)
    max_tokens = int(token_lengths.max().item())
    max_mels = int(mel_lengths.max().item())
    path = torch.zeros(batch_size, max_tokens, max_mels, dtype=torch.float32, device=token_lengths.device)
    for batch_index in range(batch_size):
        token_count = int(token_lengths[batch_index].item())
        mel_count = int(mel_lengths[batch_index].item())
        if token_count <= 0 or mel_count <= 0:
            continue
        boundaries = torch.linspace(0, mel_count, token_count + 1, device=token_lengths.device).round().long()
        for token_index in range(token_count):
            start = int(boundaries[token_index].item())
            end = int(boundaries[token_index + 1].item())
            if end <= start:
                end = min(start + 1, mel_count)
            path[batch_index, token_index, start:end] = 1.0
    return path


class ConvTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.prior_mean = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.prior_log_scale = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor, token_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del token_lengths
        encoded = self.embedding(tokens).transpose(1, 2)
        hidden = self.encoder(encoded)
        return hidden, self.prior_mean(hidden), self.prior_log_scale(hidden)


class PosteriorEncoder(nn.Module):
    def __init__(self, mel_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.posterior_mean = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.posterior_log_scale = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, mels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(mels.transpose(1, 2))
        return self.posterior_mean(hidden), self.posterior_log_scale(hidden)


class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(1)


class MelDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, mel_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, mel_dim, kernel_size=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents).transpose(1, 2)


class MiniVITS(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        mel_dim: int,
        hidden_dim: int = 192,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.text_encoder = ConvTextEncoder(vocab_size, hidden_dim, latent_dim)
        self.posterior_encoder = PosteriorEncoder(mel_dim, hidden_dim, latent_dim)
        self.duration_predictor = DurationPredictor(hidden_dim)
        self.decoder = MelDecoder(latent_dim, hidden_dim, mel_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> dict:
        hidden, prior_mean, prior_log_scale = self.text_encoder(tokens, token_lengths)
        posterior_mean, posterior_log_scale = self.posterior_encoder(mels)

        path = generate_path(token_lengths, mel_lengths)
        aligned_prior_mean = torch.matmul(prior_mean, path)
        aligned_prior_log_scale = torch.matmul(prior_log_scale, path)

        noise = torch.randn_like(posterior_mean)
        latents = posterior_mean + noise * torch.exp(posterior_log_scale)
        mel_prediction = self.decoder(latents)

        duration_targets = path.sum(dim=2)
        log_duration_targets = torch.log(duration_targets + 1.0)
        log_duration_prediction = self.duration_predictor(hidden)

        mel_mask = sequence_mask(mel_lengths, mels.size(1)).unsqueeze(-1).float()
        token_mask = sequence_mask(token_lengths, tokens.size(1)).float()

        mel_loss = F.l1_loss(mel_prediction * mel_mask, mels * mel_mask)
        duration_loss = F.mse_loss(
            log_duration_prediction * token_mask,
            log_duration_targets * token_mask,
        )
        kl_elements = (
            aligned_prior_log_scale
            - posterior_log_scale
            + (
                torch.exp(2.0 * posterior_log_scale)
                + (posterior_mean - aligned_prior_mean) ** 2
            )
            / (2.0 * torch.exp(2.0 * aligned_prior_log_scale))
            - 0.5
        )
        kl_loss = (kl_elements * mel_mask.transpose(1, 2)).sum() / mel_mask.sum().clamp_min(1.0)
        total_loss = mel_loss + 0.1 * duration_loss + 0.01 * kl_loss

        return {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "kl_loss": kl_loss,
            "mel_prediction": mel_prediction,
        }

    @torch.no_grad()
    def infer(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        hidden, prior_mean, prior_log_scale = self.text_encoder(tokens, token_lengths)
        log_durations = self.duration_predictor(hidden)
        durations = torch.clamp(torch.ceil(torch.exp(log_durations) - 1.0), min=1.0)
        durations = torch.clamp((durations * length_scale).long(), min=1)

        expanded = []
        for batch_index in range(tokens.size(0)):
            pieces = []
            token_count = int(token_lengths[batch_index].item())
            for token_index in range(token_count):
                repeat = int(durations[batch_index, token_index].item())
                mean_frame = prior_mean[batch_index : batch_index + 1, :, token_index : token_index + 1]
                log_scale_frame = prior_log_scale[batch_index : batch_index + 1, :, token_index : token_index + 1]
                noise = torch.randn(1, mean_frame.size(1), repeat, device=tokens.device)
                pieces.append(mean_frame.repeat(1, 1, repeat) + noise * torch.exp(log_scale_frame).repeat(1, 1, repeat))
            expanded.append(torch.cat(pieces, dim=2))

        max_frames = max(item.size(2) for item in expanded)
        latent_dim = expanded[0].size(1)
        latent_batch = torch.zeros(len(expanded), latent_dim, max_frames, device=tokens.device)
        for batch_index, item in enumerate(expanded):
            latent_batch[batch_index, :, : item.size(2)] = item
        return self.decoder(latent_batch)
