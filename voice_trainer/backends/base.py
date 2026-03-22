from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TrainingBackend(Protocol):
    name: str

    def prepare_training_assets(
        self,
        *,
        run_root: Path,
        selected_candidates: list[dict],
        trainer_config: dict,
    ) -> dict:
        """Prepare backend-specific dataset and config artifacts."""

    def run_training(
        self,
        *,
        asset_info: dict,
        trainer_config: dict,
    ) -> None:
        """Launch backend-specific training."""
