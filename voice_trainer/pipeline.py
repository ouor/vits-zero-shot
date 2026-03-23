from __future__ import annotations

from pathlib import Path

from .audio import load_text, write_json
from .backends import get_backend
from .config import (
    get_backend_config,
    get_training_backend_name,
    load_config,
    resolve_from_root,
)
from .generation import generate_candidates
from .logging_utils import (
    append_event,
    get_run_logger,
    log_stage_end,
    log_stage_error,
    log_stage_start,
    stage_timer,
)
from .ranking import rank_candidates
from .text_corpus import generate_prompt_items


def run_pipeline(config_path: str | Path) -> dict:
    config = load_config(config_path)
    backend_name = get_training_backend_name(config)
    backend = get_backend(backend_name)
    backend_config = get_backend_config(config, backend_name)
    reference_language = config["reference"]["language"]

    run_root = resolve_from_root(config["output_root"]) / config["run_name"]
    logger = get_run_logger(run_root)
    reference_audio = resolve_from_root(config["reference"]["audio_path"])
    reference_text = load_text(resolve_from_root(config["reference"]["text_path"]))
    stage_total = 6

    logger.info(
        "Run started | run_name=%s backend=%s language=%s output=%s",
        config["run_name"],
        backend.name,
        reference_language,
        run_root,
    )
    append_event(
        run_root,
        {
            "stage": "run",
            "status": "started",
            "config_path": str(Path(config_path).resolve()),
            "backend": backend.name,
        },
    )

    stage_name = "Prompt Generation"
    stage_index = 1
    try:
        log_stage_start(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            candidate_count=config["generation"]["candidate_count"],
        )
        with stage_timer() as elapsed:
            prompts = generate_prompt_items(
                fallback_language=reference_language,
                target_count=config["generation"]["candidate_count"],
                prompt_languages=config["generation"].get("prompt_languages"),
            )
        log_stage_end(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=elapsed(),
            prompts=len(prompts),
            artifact=run_root / "prompts.json",
        )
    except Exception as error:
        log_stage_error(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=0.0,
            error=error,
        )
        raise

    prompt_path = run_root / "prompts.json"
    write_json(prompt_path, {"prompts": prompts})

    stage_name = "Candidate Generation"
    stage_index = 2
    try:
        log_stage_start(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            prompts=len(prompts),
            model_id=config["generation"]["model_id"],
            device=config["generation"]["device"],
        )
        with stage_timer() as elapsed:
            candidates = generate_candidates(
                output_dir=run_root / "generation",
                model_id=config["generation"]["model_id"],
                device=config["generation"]["device"],
                reference_audio=reference_audio,
                reference_text=reference_text,
                prompt_items=prompts,
                temperature=config["generation"]["temperature"],
                top_k=config["generation"]["top_k"],
                repetition_penalty=config["generation"]["repetition_penalty"],
                xvec_only=config["generation"]["xvec_only"],
                non_streaming_mode=config["generation"]["non_streaming_mode"],
                max_new_tokens=config["generation"]["max_new_tokens"],
            )
        log_stage_end(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=elapsed(),
            candidates=len(candidates),
            artifact=run_root / "generation" / "candidates.jsonl",
        )
    except Exception as error:
        log_stage_error(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=0.0,
            error=error,
        )
        raise

    stage_name = "Speaker Ranking"
    stage_index = 3
    try:
        log_stage_start(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            candidates=len(candidates),
            selection_count=config["generation"]["selection_count"],
            model_source=config["speaker_ranking"]["model_source"],
        )
        with stage_timer() as elapsed:
            selected = rank_candidates(
                reference_audio=reference_audio,
                candidates=candidates,
                output_dir=run_root / "ranking",
                model_source=config["speaker_ranking"]["model_source"],
                sample_rate=config["speaker_ranking"]["embedding_sample_rate"],
                selection_count=config["generation"]["selection_count"],
                device=config["speaker_ranking"]["device"],
            )
        log_stage_end(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=elapsed(),
            selected=len(selected),
            artifact=run_root / "ranking" / "selection_summary.json",
        )
    except Exception as error:
        log_stage_error(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=0.0,
            error=error,
        )
        raise

    stage_name = "Training Asset Preparation"
    stage_index = 4
    try:
        log_stage_start(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            selected=len(selected),
            backend=backend.name,
        )
        with stage_timer() as elapsed:
            asset_info = backend.prepare_training_assets(
                run_root=run_root,
                selected_candidates=selected,
                trainer_config=backend_config,
                language=reference_language,
            )
        log_stage_end(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=elapsed(),
            dataset_dir=asset_info["dataset_dir"],
            config_path=asset_info["config_path"],
        )
    except Exception as error:
        log_stage_error(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=0.0,
            error=error,
        )
        raise

    stage_name = "Training"
    stage_index = 5
    try:
        log_stage_start(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            backend=backend.name,
            training_dir=asset_info["training_dir"],
        )
        with stage_timer() as elapsed:
            backend.run_training(asset_info=asset_info, trainer_config=backend_config)
        log_stage_end(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=elapsed(),
            training_dir=asset_info["training_dir"],
        )
    except Exception as error:
        log_stage_error(
            logger,
            run_root,
            stage=stage_name,
            index=stage_index,
            total=stage_total,
            elapsed_sec=0.0,
            error=error,
        )
        raise

    summary = {
        "run_root": str(run_root),
        "training_backend": backend.name,
        "prompt_count": len(prompts),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "training_asset_dir": asset_info["dataset_dir"],
        "trainer_config_path": asset_info["config_path"],
    }
    stage_name = "Run Finalization"
    stage_index = 6
    log_stage_start(
        logger,
        run_root,
        stage=stage_name,
        index=stage_index,
        total=stage_total,
    )
    with stage_timer() as elapsed:
        write_json(run_root / "run_summary.json", summary)
    log_stage_end(
        logger,
        run_root,
        stage=stage_name,
        index=stage_index,
        total=stage_total,
        elapsed_sec=elapsed(),
        summary_path=run_root / "run_summary.json",
        selected=len(selected),
    )
    append_event(
        run_root,
        {
            "stage": "run",
            "status": "completed",
            "summary_path": str(run_root / "run_summary.json"),
        },
    )
    logger.info(
        "Run completed | prompts=%s candidates=%s selected=%s summary=%s",
        len(prompts),
        len(candidates),
        len(selected),
        run_root / "run_summary.json",
    )
    return summary
