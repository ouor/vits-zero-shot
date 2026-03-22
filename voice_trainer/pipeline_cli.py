from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-sample voice training pipeline.")
    parser.add_argument("--config", required=True, help="Path to the pipeline JSON config.")
    args = parser.parse_args()

    summary = run_pipeline(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
