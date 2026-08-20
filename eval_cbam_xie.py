import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support,classification_report
import timm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class EfficientNetV2WithCBAM(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', num_classes=102, cbam_layers=None):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.cbam_layers = cbam_layers or []
        self.cbam_modules = nn.ModuleDict()

        for name, module in self.backbone.named_modules():
            if any(name.endswith(key) for key in self.cbam_layers):
                if isinstance(module, nn.Conv2d):
                    self.cbam_modules[name.replace('.', '_')] = CBAM(module.out_channels)

    def forward(self, x):
        for name, module in self.backbone.named_children():
            x = module(x)
            cbam_name = name.replace('.', '_')
            if cbam_name in self.cbam_modules:
                x = self.cbam_modules[cbam_name](x)
        return x

class IP102Dataset2(torch.utils.data.Dataset):
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

class DoDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        """
        Args:
            annotation_file (str): Path to CSV containing ['filepath', 'label']
            image_dir (str): Root directory of images
            transform (callable, optional): torchvision transforms
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load CSV
        self.data = pd.read_csv(annotation_file)

        if 'filepath' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("CSV must contain at least 'filepath' and 'label' columns")

        # Encode string labels to integers
        self.le = LabelEncoder()
        self.data['label_encoded'] = self.le.fit_transform(self.data['label'])
        self.classes = self.le.classes_  # store class names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['filepath']
        label = int(row['label_encoded'])

        # Make sure path is absolute
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_dir, image_path)

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


class IP102Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, keep_labels=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        samples = [line.strip().split() for line in lines]

        if keep_labels is not None:
            samples = [s for s in samples if int(s[1]) in keep_labels]

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)



if __name__ == "__main__":
    with open("train_config_xie.yaml", 'r') as file:
        config = yaml.safe_load(file)

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]
    print("model name", config["model_name"], "Input size:", input_size)
    print("Using CBAM model for evaluation...")
    # Define the transformation for the test dataset
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    #target_labels = [15, 22, 24, 39, 50, 51, 58, 60, 62, 80]
    target_labels=None
    test_dataset = DoDataset(config['test_annotation'], config['image_dir'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2WithCBAM(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        cbam_layers=config['cbam_layers']
    )
    model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'aug_m_cbam_best_model_xie.pth'), map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * config['num_classes']
    class_total = [0] * config['num_classes']
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

    print("\nPer-Class Accuracy:")
    per_class_stats = []
    for i in range(config['num_classes']):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            per_class_stats.append({"class": i, "accuracy": acc, "correct": class_correct[i], "total": class_total[i]})
            print(f"Class {i:3d}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(config['num_classes'])), zero_division=0)
    for i in range(config['num_classes']):
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
    df_stats.to_csv(os.path.join(config['log_dir'], "cbam_m_per_class_metrics.csv"), index=False)


