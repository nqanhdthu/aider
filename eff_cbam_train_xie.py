import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import timm
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


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


class IP102Dataset2(Dataset):
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


class IP102Dataset(Dataset):
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


if __name__ == "__main__":
    with open("train_config_xie.yaml", 'r') as file:
        config = yaml.safe_load(file)

    os.makedirs(config['log_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=config['log_dir'])
    early_stop_patience = config.get('early_stop_patience', 10)
    early_stop_counter = 0

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]
    print("model name", config["model_name"], "Input size:", input_size)

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),  # zoom_range = 0.1
        transforms.RandomAffine(
            degrees=25,  # rotation_range = 25
            translate=(0.1, 0.1),  # height_shift_range = 0.1 (used for both H & W)
            shear=20,  # shear_range = 0.2 (converted to degrees)
            fill=0  # fill_mode = "nearest" equivalent (zero fill is default)
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    #target_labels = [15, 22, 24, 39, 50, 51, 58, 60, 62, 80]
    target_labels=None
    train_dataset = DoDataset(config['train_annotation'], config['image_dir'], transform=train_transform)
    val_dataset = DoDataset(config['val_annotation'], config['image_dir'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = EfficientNetV2WithCBAM(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        cbam_layers=config['cbam_layers']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch+1)

        model.eval()
        #torch.cuda.empty_cache()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        writer.add_scalar('Accuracy/val', acc, epoch+1)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(config['log_dir'], 'aug_m_cbam_best_model_xie.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        scheduler.step()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")

    writer.close()
