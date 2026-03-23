from __future__ import annotations

import gc
from pathlib import Path

from tqdm.auto import tqdm

from .audio import save_waveform, write_jsonl


def generate_candidates(
    *,
    output_dir: Path,
    model_id: str,
    device: str,
    reference_audio: Path,
    reference_text: str,
    prompt_items: list[dict],
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    xvec_only: bool,
    non_streaming_mode: bool,
    max_new_tokens: int,
) -> list[dict]:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS  # pylint: disable=import-error

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    model = FasterQwen3TTS.from_pretrained(
        model_id,
        device=device,
        dtype=torch.bfloat16,
    )
    manifest = []

    try:
        progress = tqdm(prompt_items, desc="Candidate Generation", unit="utt")
        for index, prompt in enumerate(progress, start=1):
            text = prompt["text"]
            language = prompt["language"]
            audio_list, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(reference_audio),
                ref_text=reference_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
            )

            item_id = f"candidate_{index:04d}"
            wav_path = wav_dir / f"{item_id}.wav"
            save_waveform(wav_path, audio_list[0], sample_rate)
            progress.set_postfix_str(item_id)
            manifest.append(
                {
                    "id": item_id,
                    "text": text,
                    "language": language,
                    "wav_path": str(wav_path),
                }
            )
    finally:
        del model
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(output_dir / "candidates.jsonl", manifest)
    return manifest
