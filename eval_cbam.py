import argparse
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from eff_cbam import EfficientNetV2WithCBAM, IP102Dataset


def evaluate_cbam(config_path="train_config.yaml", checkpoint_name="aug_m_cbam_best_model.pth"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    input_key = config["input_key"]
    input_size = config["input_size"][input_key]
    print("model name", config["model_name"], "Input size:", input_size)
    print("Using CBAM model for evaluation...")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    test_dataset = IP102Dataset(
        config["test_annotation"],
        config["image_dir"],
        transform=transform,
        keep_labels=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2WithCBAM(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        cbam_layers=config["cbam_layers"],
    )
    checkpoint_path = os.path.join(config["log_dir"], checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * config["num_classes"]
    class_total = [0] * config["num_classes"]
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating CBAM model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    overall_acc = 100 * correct / total
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(config["num_classes"])),
        average=None,
        zero_division=0,
    )

    print("\nPer-Class Accuracy:")
    per_class_stats = []
    for i in range(config["num_classes"]):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        per_class_stats.append({
            "class": i,
            "accuracy": acc,
            "correct": class_correct[i],
            "total": class_total[i],
            "precision": precision[i] * 100,
            "recall": recall[i] * 100,
            "f1_score": f1[i] * 100,
            "support": support[i],
        })
        if class_total[i] > 0:
            print(f"Class {i:3d}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    macro_precision = precision.mean() * 100
    macro_recall = recall.mean() * 100
    macro_f1 = f1.mean() * 100

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    weighted_precision *= 100
    weighted_recall *= 100
    weighted_f1 *= 100

    print(f"\nMacro-Averaged Precision: {macro_precision:.2f}%")
    print(f"Macro-Averaged Recall:    {macro_recall:.2f}%")
    print(f"Macro-Averaged F1-score:  {macro_f1:.2f}%")
    print(f"Weighted Precision:       {weighted_precision:.2f}%")
    print(f"Weighted Recall:          {weighted_recall:.2f}%")
    print(f"Weighted F1-score:        {weighted_f1:.2f}%")

    df_stats = pd.DataFrame(per_class_stats)
    df_stats.to_csv(os.path.join(config["log_dir"], "cbam_m_per_class_metrics.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CBAM model using shared train components.")
    parser.add_argument(
        "--config",
        default="train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="aug_m_cbam_best_model.pth",
        help="Checkpoint file name inside config['log_dir']",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_cbam(config_path=args.config, checkpoint_name=args.checkpoint)
