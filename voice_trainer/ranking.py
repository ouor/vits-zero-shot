from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier

from .audio import load_waveform, write_json, write_jsonl


def _compute_embedding(
    classifier: EncoderClassifier,
    audio_path: Path,
    sample_rate: int,
) -> torch.Tensor:
    waveform, _ = load_waveform(audio_path, sample_rate=sample_rate)
    embedding = classifier.encode_batch(waveform)
    return embedding.squeeze().detach().cpu()


def rank_candidates(
    *,
    reference_audio: Path,
    candidates: list[dict],
    output_dir: Path,
    model_source: str,
    sample_rate: int,
    selection_count: int,
    device: str = "cpu",
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        run_opts={"device": device},
        savedir=str(output_dir / "pretrained_speaker_model"),
    )

    reference_embedding = _compute_embedding(classifier, reference_audio, sample_rate)
    ranked = []
    for candidate in candidates:
        candidate_embedding = _compute_embedding(
            classifier,
            Path(candidate["wav_path"]),
            sample_rate,
        )
        score = F.cosine_similarity(
            reference_embedding.unsqueeze(0),
            candidate_embedding.unsqueeze(0),
        ).item()
        ranked.append({**candidate, "speaker_similarity": score})

    ranked.sort(key=lambda item: item["speaker_similarity"], reverse=True)
    selected = ranked[:selection_count]

    write_jsonl(output_dir / "ranked_candidates.jsonl", ranked)
    write_json(
        output_dir / "selection_summary.json",
        {
            "reference_audio": str(reference_audio),
            "candidate_count": len(candidates),
            "selection_count": len(selected),
            "top_score": selected[0]["speaker_similarity"] if selected else None,
            "bottom_selected_score": selected[-1]["speaker_similarity"] if selected else None,
        },
    )
    return selected
