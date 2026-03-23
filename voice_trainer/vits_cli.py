from __future__ import annotations

import argparse
import json

from .vits_train import train_vits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local compact VITS-style model.")
    parser.add_argument("--config", required=True, help="Path to the VITS JSON config.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    args = parser.parse_args()

    summary = train_vits(args.config, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
