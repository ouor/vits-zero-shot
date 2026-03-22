from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from .audio import write_json
from .vits.runtime import ensure_monotonic_align_built


def run_training_command(
    *,
    training_command: list[str],
    config_path: Path,
    output_dir: Path,
) -> None:
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
    subprocess.run(resolved_command, check=True)
