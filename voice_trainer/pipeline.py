from __future__ import annotations

from pathlib import Path

from .audio import load_text, write_json
from .config import load_config, resolve_from_root
from .generation import generate_candidates
from .ranking import rank_candidates
from .text_corpus import generate_sentences
from .training import run_training_command
from .vits_dataset import build_vits_config, export_vits_dataset


def run_pipeline(config_path: str | Path) -> dict:
    config = load_config(config_path)

    run_root = resolve_from_root(config["output_root"]) / config["run_name"]
    reference_audio = resolve_from_root(config["reference"]["audio_path"])
    reference_text = load_text(resolve_from_root(config["reference"]["text_path"]))

    prompts = generate_sentences(
        config["reference"]["language"],
        config["generation"]["candidate_count"],
    )
    prompt_path = run_root / "prompts.json"
    write_json(prompt_path, {"prompts": prompts})

    candidates = generate_candidates(
        output_dir=run_root / "generation",
        model_id=config["generation"]["model_id"],
        device=config["generation"]["device"],
        reference_audio=reference_audio,
        reference_text=reference_text,
        language=config["reference"]["language"],
        texts=prompts,
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
        repetition_penalty=config["generation"]["repetition_penalty"],
        xvec_only=config["generation"]["xvec_only"],
        non_streaming_mode=config["generation"]["non_streaming_mode"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )

    selected = rank_candidates(
        reference_audio=reference_audio,
        candidates=candidates,
        output_dir=run_root / "ranking",
        model_source=config["speaker_ranking"]["model_source"],
        sample_rate=config["speaker_ranking"]["embedding_sample_rate"],
        selection_count=config["generation"]["selection_count"],
        device=config["speaker_ranking"]["device"],
    )

    dataset_info = export_vits_dataset(
        selected_candidates=selected,
        output_dir=run_root / "vits_data",
        target_sample_rate=config["vits"]["target_sample_rate"],
        train_split_ratio=config["vits"]["train_split_ratio"],
    )

    vits_config_path = run_root / "vits_data" / "vits_config.json"
    build_vits_config(
        output_path=vits_config_path,
        train_filelist=dataset_info["train_filelist"],
        val_filelist=dataset_info["val_filelist"],
        batch_size=config["vits"]["batch_size"],
        epochs=config["vits"]["epochs"],
        sampling_rate=config["vits"]["target_sample_rate"],
        pretrained_generator=config["vits"].get("pretrained_generator", ""),
        pretrained_discriminator=config["vits"].get("pretrained_discriminator", ""),
    )

    run_training_command(
        training_command=config["vits"]["training_command"],
        config_path=vits_config_path,
        output_dir=run_root / "training",
    )

    summary = {
        "run_root": str(run_root),
        "prompt_count": len(prompts),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "vits_config_path": str(vits_config_path),
    }
    write_json(run_root / "run_summary.json", summary)
    return summary
