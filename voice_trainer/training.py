from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .audio import write_json
from .vits_train import train_vits


def run_training_command(
    *,
    training_command: list[str],
    config_path: Path,
    output_dir: Path,
) -> None:
    if not training_command:
        summary = train_vits(config_path, output_dir)
        write_json(output_dir / "training_summary_wrapper.json", summary)
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
