from __future__ import annotations

from pathlib import Path

from ..datasets import export_selected_corpus
from ..training import run_training_command
from .vits_preparation import build_vits_config


class VitsBackend:
    name = "vits"

    def prepare_training_assets(
        self,
        *,
        run_root: Path,
        selected_candidates: list[dict],
        trainer_config: dict,
    ) -> dict:
        vits_data_dir = run_root / "vits_data"
        dataset_info = export_selected_corpus(
            selected_candidates=selected_candidates,
            output_dir=vits_data_dir,
            target_sample_rate=trainer_config["target_sample_rate"],
            train_split_ratio=trainer_config["train_split_ratio"],
        )

        config_path = vits_data_dir / "vits_config.json"
        build_vits_config(
            output_path=config_path,
            train_filelist=dataset_info["train_filelist"],
            val_filelist=dataset_info["val_filelist"],
            batch_size=trainer_config["batch_size"],
            epochs=trainer_config["epochs"],
            sampling_rate=trainer_config["target_sample_rate"],
            pretrained_generator=trainer_config.get("pretrained_generator", ""),
            pretrained_discriminator=trainer_config.get("pretrained_discriminator", ""),
        )

        return {
            "backend": self.name,
            "dataset_dir": str(vits_data_dir),
            "config_path": str(config_path),
            "training_dir": str(run_root / "training"),
            "dataset_info": dataset_info,
        }

    def run_training(
        self,
        *,
        asset_info: dict,
        trainer_config: dict,
    ) -> None:
        run_training_command(
            training_command=trainer_config.get("training_command", []),
            config_path=Path(asset_info["config_path"]),
            output_dir=Path(asset_info["training_dir"]),
        )
