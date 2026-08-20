from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import FrameworkConfig


@dataclass(frozen=True)
class Sample:
    image_path: str
    label_raw: str


def _resolve_image_path(image_dir: str, image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_dir, image_path)


def load_samples(annotation_path: str, image_dir: str) -> list[Sample]:
    if annotation_path.lower().endswith(".txt"):
        samples: list[Sample] = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                image_path, label = parts
                samples.append(Sample(_resolve_image_path(image_dir, image_path), str(label)))
        return samples

    df = pd.read_csv(annotation_path)
    if "filepath" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{annotation_path} must contain filepath,label columns")

    samples = []
    for _, row in df.iterrows():
        image_path = str(row["filepath"])
        label = str(row["label"])
        samples.append(Sample(_resolve_image_path(image_dir, image_path), label))
    return samples


def _sort_labels(labels: set[str]) -> list[str]:
    try:
        return [str(x) for x in sorted(labels, key=lambda x: int(x))]
    except ValueError:
        return sorted(labels)


def build_label_mapping(*splits: list[Sample]) -> dict[str, int]:
    labels = {sample.label_raw for split in splits for sample in split}
    ordered = _sort_labels(labels)
    return {label: idx for idx, label in enumerate(ordered)}


class UnifiedImageDataset(Dataset):
    def __init__(self, samples: list[Sample], label_to_idx: dict[str, int], transform: Any = None):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[sample.label_raw]
        return image, label


def train_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
        transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), shear=20, fill=0),
        transforms.ToTensor(),
    ])


def eval_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])


def save_label_mapping(path: str, label_to_idx: dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=True, indent=2)


def load_label_mapping(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def build_dataloaders(
    cfg: FrameworkConfig,
    use_saved_labels: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    train_samples = load_samples(cfg.train_annotation, cfg.image_dir)
    val_samples = load_samples(cfg.val_annotation, cfg.image_dir)
    test_samples = load_samples(cfg.test_annotation, cfg.image_dir)

    if use_saved_labels and os.path.exists(cfg.labels_path):
        label_to_idx = load_label_mapping(cfg.labels_path)
    else:
        label_to_idx = build_label_mapping(train_samples, val_samples, test_samples)

    train_ds = UnifiedImageDataset(train_samples, label_to_idx, train_transform(cfg.input_size))
    val_ds = UnifiedImageDataset(val_samples, label_to_idx, eval_transform(cfg.input_size))
    test_ds = UnifiedImageDataset(test_samples, label_to_idx, eval_transform(cfg.input_size))

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "train_val": DataLoader(
            UnifiedImageDataset(train_samples + val_samples, label_to_idx, eval_transform(cfg.input_size)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }
    return loaders, label_to_idx
