from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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


def _to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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


def _build_feature_classifier(classifier_type: str):
    normalized = classifier_type.strip().lower()
    if normalized == "svm":
        return (
            "SVM",
            SVC(random_state=42),
            {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
            True,
        )

    if normalized == "knn":
        return (
            "KNN",
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
            True,
        )

    if normalized in {"logistic", "logistic_regression", "lr"}:
        return (
            "LogisticRegression",
            LogisticRegression(random_state=42, max_iter=2000),
            [
                {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.1, 1, 10]},
                {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.1, 1, 10]},
            ],
            True,
        )

    if normalized in {"decision_tree", "dt"}:
        return (
            "DecisionTree",
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            },
            False,
        )

    if normalized in {"random_forest", "rf"}:
        return (
            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 20, 40],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
            False,
        )

    if normalized in {"xgboost", "xgb"}:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it to use classifier_type='xgboost'."
            ) from exc
        return (
            "XGBoost",
            xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
            {"n_estimators": [200, 400], "max_depth": [4, 8], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
            False,
        )

    if normalized in {"lightgbm", "lgbm", "lgb"}:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it to use classifier_type='lightgbm'."
            ) from exc
        return (
            "LightGBM",
            lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            {"n_estimators": [200, 400], "max_depth": [-1, 20], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
            False,
        )

    if normalized in {"catboost", "cat"}:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "catboost is not installed. Install it to use classifier_type='catboost'."
            ) from exc
        return (
            "CatBoost",
            CatBoostClassifier(random_state=42, verbose=False),
            {"iterations": [300, 600], "depth": [6, 8], "learning_rate": [0.05, 0.1]},
            False,
        )

    raise ValueError(
        "Unsupported classifier_type: "
        f"{classifier_type}. Supported: svm, knn, logistic_regression, decision_tree, random_forest, xgboost, lightgbm, catboost."
    )


def run_feature_classification_pipeline(
    cfg: FrameworkConfig,
    model_type: str,
    checkpoint_path: str | None = None,
    classifier_type: str = "svm",
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

    classifier_name, classifier, param_grid, use_scaler = _build_feature_classifier(classifier_type)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train_features)
    x_test_scaled = scaler.transform(test_features)
    x_train = x_train_scaled if use_scaler else train_features
    x_test = x_test_scaled if use_scaler else test_features

    grid = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(x_train, train_labels)
    best_model = grid.best_estimator_

    train_pred = best_model.predict(x_train)
    test_pred = best_model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        test_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    class_report = classification_report(test_labels, test_pred, digits=4, output_dict=True)
    cm = confusion_matrix(test_labels, test_pred)
    metrics = {
        "classifier_type": classifier_type,
        "classifier_name": classifier_name,
        "best_params": _to_builtin(grid.best_params_),
        "cv_accuracy": float(grid.best_score_),
        "train_accuracy": float(accuracy_score(train_labels, train_pred)),
        "test_accuracy": float(accuracy_score(test_labels, test_pred)),
        "macro_precision": float(precision.mean() * 100.0),
        "macro_recall": float(recall.mean() * 100.0),
        "macro_f1": float(f1.mean() * 100.0),
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
        "checkpoint_path": checkpoint_path,
    }
    metrics = _to_builtin(metrics)

    classifier_key = classifier_type.strip().lower()
    with open(os.path.join(cfg.output_dir, f"{classifier_key}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    with open(os.path.join(cfg.output_dir, f"{classifier_key}_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    if use_scaler:
        with open(os.path.join(cfg.output_dir, f"{classifier_key}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # Backward compatibility for legacy SVM artifact names.
    if classifier_key == "svm":
        with open(os.path.join(cfg.output_dir, "svm_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=2)
        with open(os.path.join(cfg.output_dir, "svm_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        with open(os.path.join(cfg.output_dir, "svm_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    return metrics


def run_feature_svm_pipeline(
    cfg: FrameworkConfig,
    model_type: str,
    checkpoint_path: str | None = None,
) -> dict:
    return run_feature_classification_pipeline(
        cfg=cfg,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        classifier_type="svm",
    )


def run_full_pipeline(cfg: FrameworkConfig, model_type: str, classifier_type: str = "svm") -> dict:
    train_result = train_pipeline(cfg, model_type=model_type)
    eval_metrics = evaluate_pipeline(cfg, model_type=model_type, checkpoint_path=train_result.checkpoint_path)
    classifier_metrics = run_feature_classification_pipeline(
        cfg,
        model_type=model_type,
        checkpoint_path=train_result.checkpoint_path,
        classifier_type=classifier_type,
    )
    return {
        "checkpoint_path": train_result.checkpoint_path,
        "best_val_accuracy": train_result.best_val_accuracy,
        "eval": eval_metrics,
        "feature_classifier": classifier_metrics,
    }
