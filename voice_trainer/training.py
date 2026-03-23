from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .audio import write_json


def run_training_command(
    *,
    training_command: list[str],
    config_path: Path,
    output_dir: Path,
) -> None:
    if not training_command:
        write_json(
            output_dir / "training_skipped.json",
            {
                "reason": "No training command configured.",
                "config_path": str(config_path),
            },
        )
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
