import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import yaml
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np

# Dataset
class IP102Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        self.samples = [line.strip().split() for line in lines if len(line.strip().split()) == 2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)

# ECA Module
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

# EfficientNetV2 + ECA
class EfficientNetV2WithECA(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', num_classes=102, eca_layers=None):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.eca_layers = eca_layers or []
        self.eca_modules = nn.ModuleDict()

        named_modules = list(self.backbone.named_modules())
        for name, module in named_modules:
            if any(name.endswith(key) for key in self.eca_layers):
                if isinstance(module, nn.Conv2d):
                    self.eca_modules[name.replace('.', '_')] = ECABlock(module.out_channels)

    def forward(self, x):
        for name, module in self.backbone.named_children():
            x = module(x)
            eca_name = name.replace('.', '_')
            if eca_name in self.eca_modules:
                x = self.eca_modules[eca_name](x)
        return x

if __name__ == "__main__":
    with open("train_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    input_size = 300 if 's' in config['model_name'] else 384
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    test_dataset = IP102Dataset(config['test_annotation'], config['image_dir'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2WithECA(
        model_name=config['model_name'],
        num_classes=102,
        eca_layers=config['eca_layers']
    )
    model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'aug_m_eca_best_model.pth'), map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * 102
    class_total = [0] * 102
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating ECA model"):
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

    # After computing all_labels and all_preds
    report = classification_report(all_labels, all_preds, labels=list(range(102)), digits=4)
    print("\nClassification Report:")
    print(report)

    # Optionally save to a file
    with open(os.path.join(config['log_dir'], "classification_report.txt"), "w") as f:
        f.write(report)

    print("\nPer-Class Accuracy:")
    per_class_stats = []
    for i in range(102):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            per_class_stats.append({"class": i, "accuracy": acc, "correct": class_correct[i], "total": class_total[i]})
            print(f"Class {i:3d}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(102)), zero_division=0)
    for i in range(102):
        per_class_stats[i]["precision"] = precision[i] * 100
        per_class_stats[i]["recall"] = recall[i] * 100
        per_class_stats[i]["f1_score"] = f1[i] * 100

    macro_precision = precision.mean() * 100
    macro_recall = recall.mean() * 100
    macro_f1 = f1.mean() * 100

    print(f"\nMacro-Averaged Precision: {macro_precision:.2f}%")
    print(f"Macro-Averaged Recall:    {macro_recall:.2f}%")
    print(f"Macro-Averaged F1-score:  {macro_f1:.2f}%")

    df_stats = pd.DataFrame(per_class_stats)
    df_stats.to_csv(os.path.join(config['log_dir'], "eca_m_aug_per_class_metrics.csv"), index=False)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(102)))
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title("ECA Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], "eca_confusion_matrix.png"))
    print("Confusion matrix saved.")