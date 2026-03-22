from __future__ import annotations

from pathlib import Path

from .audio import load_text, write_json
from .backends import get_backend
from .config import load_config, resolve_from_root
from .generation import generate_candidates
from .ranking import rank_candidates
from .text_corpus import generate_sentences


def run_pipeline(config_path: str | Path) -> dict:
    config = load_config(config_path)
    backend_name = config.get("training_backend", "vits")
    backend = get_backend(backend_name)

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

    asset_info = backend.prepare_training_assets(
        run_root=run_root,
        selected_candidates=selected,
        trainer_config=config["vits"],
    )
    backend.run_training(asset_info=asset_info, trainer_config=config["vits"])

    summary = {
        "run_root": str(run_root),
        "training_backend": backend.name,
        "prompt_count": len(prompts),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "training_asset_dir": asset_info["dataset_dir"],
        "trainer_config_path": asset_info["config_path"],
    }
    write_json(run_root / "run_summary.json", summary)
    return summary
