from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from .config import FrameworkConfig
from .data import build_dataloaders, save_label_mapping
from .modeling import build_model, extract_backbone_features, load_checkpoint


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: str
    best_val_accuracy: float


def _default_checkpoint_name(model_type: str) -> str:
    return f"best_{model_type}.pth"


def _evaluate_on_loader(model: nn.Module, loader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    accuracy = (100.0 * correct / total) if total else 0.0
    return accuracy, all_labels, all_preds


def train_pipeline(cfg: FrameworkConfig, model_type: str, checkpoint_name: str | None = None) -> TrainResult:
    os.makedirs(cfg.output_dir, exist_ok=True)
    loaders, label_to_idx = build_dataloaders(cfg)
    save_label_mapping(cfg.labels_path, label_to_idx)

    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type=model_type,
        model_name=cfg.model_name,
        num_classes=num_classes,
        cbam_layers=cfg.cbam_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    early_stop_counter = 0
    checkpoint_name = checkpoint_name or _default_checkpoint_name(model_type)
    checkpoint_path = os.path.join(cfg.output_dir, checkpoint_name)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, _, _ = _evaluate_on_loader(model, loaders["val"], device)
        avg_loss = running_loss / max(len(loaders["train"]), 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

    return TrainResult(checkpoint_path=checkpoint_path, best_val_accuracy=best_val_acc)


def evaluate_pipeline(cfg: FrameworkConfig, model_type: str, checkpoint_path: str | None = None) -> dict:
    loaders, label_to_idx = build_dataloaders(cfg, use_saved_labels=True)
    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type=model_type,
        model_name=cfg.model_name,
        num_classes=num_classes,
        cbam_layers=cfg.cbam_layers,
    ).to(device)

    checkpoint_path = checkpoint_path or os.path.join(cfg.output_dir, _default_checkpoint_name(model_type))
    load_checkpoint(model, checkpoint_path, device)

    test_acc, all_labels, all_preds = _evaluate_on_loader(model, loaders["test"], device)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    macro_precision = float(precision.mean() * 100.0)
    macro_recall = float(recall.mean() * 100.0)
    macro_f1 = float(f1.mean() * 100.0)

    metrics = {
        "accuracy": float(test_acc),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "checkpoint_path": checkpoint_path,
        "num_classes": num_classes,
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    per_class = pd.DataFrame(
        {
            "class_idx": list(range(num_classes)),
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
            "support": support,
        }
    )
    per_class.to_csv(os.path.join(cfg.output_dir, "eval_per_class.csv"), index=False)
    return metrics


def _collect_features_and_labels(model: nn.Module, loader, device: torch.device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            feats = extract_backbone_features(model, images)
            features.append(feats.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def run_feature_svm_pipeline(
    cfg: FrameworkConfig,
    model_type: str,
    checkpoint_path: str | None = None,
) -> dict:
    loaders, label_to_idx = build_dataloaders(cfg, use_saved_labels=True)
    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type=model_type,
        model_name=cfg.model_name,
        num_classes=num_classes,
        cbam_layers=cfg.cbam_layers,
    ).to(device)

    checkpoint_path = checkpoint_path or os.path.join(cfg.output_dir, _default_checkpoint_name(model_type))
    load_checkpoint(model, checkpoint_path, device)

    train_features, train_labels = _collect_features_and_labels(model, loaders["train_val"], device)
    test_features, test_labels = _collect_features_and_labels(model, loaders["test"], device)

    feature_dir = os.path.join(cfg.output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    np.save(os.path.join(feature_dir, "train_val_features.npy"), train_features)
    np.save(os.path.join(feature_dir, "train_val_labels.npy"), train_labels)
    np.save(os.path.join(feature_dir, "test_features.npy"), test_features)
    np.save(os.path.join(feature_dir, "test_labels.npy"), test_labels)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train_features)
    x_test_scaled = scaler.transform(test_features)

    grid = GridSearchCV(
        SVC(random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(x_train_scaled, train_labels)
    svm = grid.best_estimator_

    train_pred = svm.predict(x_train_scaled)
    test_pred = svm.predict(x_test_scaled)
    metrics = {
        "svm_best_params": grid.best_params_,
        "svm_cv_accuracy": float(grid.best_score_),
        "svm_train_accuracy": float(accuracy_score(train_labels, train_pred)),
        "svm_test_accuracy": float(accuracy_score(test_labels, test_pred)),
        "checkpoint_path": checkpoint_path,
    }

    with open(os.path.join(cfg.output_dir, "svm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    with open(os.path.join(cfg.output_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)
    with open(os.path.join(cfg.output_dir, "svm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    return metrics


def run_full_pipeline(cfg: FrameworkConfig, model_type: str) -> dict:
    train_result = train_pipeline(cfg, model_type=model_type)
    eval_metrics = evaluate_pipeline(cfg, model_type=model_type, checkpoint_path=train_result.checkpoint_path)
    svm_metrics = run_feature_svm_pipeline(cfg, model_type=model_type, checkpoint_path=train_result.checkpoint_path)
    return {
        "checkpoint_path": train_result.checkpoint_path,
        "best_val_accuracy": train_result.best_val_accuracy,
        "eval": eval_metrics,
        "svm": svm_metrics,
    }
