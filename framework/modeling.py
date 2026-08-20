from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm

from eff_cbam import EfficientNetV2WithCBAM


def build_model(
    model_type: str,
    model_name: str,
    num_classes: int,
    cbam_layers: list[str] | None = None,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "baseline":
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if model_type == "cbam":
        return EfficientNetV2WithCBAM(
            model_name=model_name,
            num_classes=num_classes,
            cbam_layers=cbam_layers or [],
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state: Any = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError:
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(filtered)
        model.load_state_dict(model_state)


def extract_backbone_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    feature_model = model.backbone if hasattr(model, "backbone") else model
    if not hasattr(feature_model, "forward_features"):
        raise ValueError("Model does not support forward_features for backbone extraction.")
    features = feature_model.forward_features(images)
    if features.ndim == 4:
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
    elif features.ndim > 2:
        features = features.flatten(1)
    return features
