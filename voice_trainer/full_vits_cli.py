from __future__ import annotations

import argparse
import subprocess
import sys

from .full_vits.runtime import ensure_monotonic_align_built


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the vendored full VITS implementation.")
    parser.add_argument("--config", required=True, help="Path to the VITS JSON config.")
    parser.add_argument("--model-dir", required=True, help="Output directory for checkpoints and logs.")
    args = parser.parse_args()

    ensure_monotonic_align_built()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "voice_trainer.full_vits.train",
            "-c",
            args.config,
            "-m",
            args.model_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
