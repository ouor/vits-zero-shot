from __future__ import annotations

import gc
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from .audio import write_json
from .vits.runtime import ensure_monotonic_align_built

logger = logging.getLogger(__name__)


def _release_parent_process_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            "Released parent CUDA cache | allocated_mb=%.1f reserved_mb=%.1f",
            torch.cuda.memory_allocated() / (1024 * 1024),
            torch.cuda.memory_reserved() / (1024 * 1024),
        )


def run_training_command(
    *,
    training_command: list[str],
    config_path: Path,
    output_dir: Path,
) -> None:
    _release_parent_process_memory()
    if not training_command:
        ensure_monotonic_align_built()
        default_command = [
            sys.executable,
            "-m",
            "voice_trainer.vits.train",
            "-c",
            str(config_path),
            "-m",
            str(output_dir),
        ]
        write_json(
            output_dir / "training_command.json",
            {
                "command": default_command,
                "shell_preview": " ".join(shlex.quote(part) for part in default_command),
            },
        )
        logger.info(
            "Training command prepared | config=%s output_dir=%s command=%s",
            config_path,
            output_dir,
            " ".join(shlex.quote(part) for part in default_command),
        )
        subprocess.run(default_command, check=True)
        return

    resolved_command = [
        part.format(config_path=str(config_path), output_dir=str(output_dir))
        for part in training_command
    ]
    write_json(
        output_dir / "training_command.json",
        {
            "command": resolved_command,
            "shell_preview": " ".join(shlex.quote(part) for part in resolved_command),
        },
    )
    logger.info(
        "Training command prepared | config=%s output_dir=%s command=%s",
        config_path,
        output_dir,
        " ".join(shlex.quote(part) for part in resolved_command),
    )
    subprocess.run(resolved_command, check=True)
