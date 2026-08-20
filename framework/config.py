from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILES = {
    "ip102": "train_config.yaml",
    "do": "train_config_Do.yaml",
    "xie": "train_config_xie.yaml",
}


@dataclass(frozen=True)
class FrameworkConfig:
    dataset_name: str
    config_path: str
    train_annotation: str
    val_annotation: str
    test_annotation: str
    image_dir: str
    batch_size: int
    learning_rate: float
    epochs: int
    model_name: str
    cbam_layers: list[str]
    input_size: int
    log_dir: str
    early_stop_patience: int
    num_workers: int

    @property
    def output_dir(self) -> str:
        return os.path.join(self.log_dir, "framework", self.dataset_name)

    @property
    def labels_path(self) -> str:
        return os.path.join(self.output_dir, "label_mapping.json")


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_config_path(dataset_name: str, config_path: str | None) -> str:
    if config_path:
        return str(Path(config_path))
    if dataset_name not in DEFAULT_CONFIG_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return str(Path(DEFAULT_CONFIG_FILES[dataset_name]))


def load_framework_config(dataset_name: str, config_path: str | None = None) -> FrameworkConfig:
    resolved_path = _resolve_config_path(dataset_name, config_path)
    raw = _read_yaml(resolved_path)

    input_size_cfg = raw.get("input_size", {})
    input_key = raw.get("input_key", "m")
    if isinstance(input_size_cfg, dict):
        input_size = int(input_size_cfg.get(input_key, 384))
    else:
        input_size = int(input_size_cfg)

    cbam_layers_raw = raw.get("cbam_layers", [])
    if isinstance(cbam_layers_raw, str):
        cbam_layers = [cbam_layers_raw]
    elif cbam_layers_raw is None:
        cbam_layers = []
    else:
        cbam_layers = [str(x) for x in cbam_layers_raw]

    return FrameworkConfig(
        dataset_name=dataset_name,
        config_path=resolved_path,
        train_annotation=str(raw["train_annotation"]),
        val_annotation=str(raw["val_annotation"]),
        test_annotation=str(raw["test_annotation"]),
        image_dir=str(raw["image_dir"]),
        batch_size=int(raw.get("batch_size", 16)),
        learning_rate=float(raw.get("learning_rate", 5e-4)),
        epochs=int(raw.get("epochs", 30)),
        model_name=str(raw.get("model_name", "tf_efficientnetv2_m")),
        cbam_layers=cbam_layers,
        input_size=input_size,
        log_dir=str(raw.get("log_dir", "logs")),
        early_stop_patience=int(raw.get("early_stop_patience", 10)),
        num_workers=int(raw.get("num_workers", 4)),
    )
