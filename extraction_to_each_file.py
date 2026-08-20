import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import numpy as np
from tqdm import tqdm
import timm


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
    def __init__(self, model_name='tf_efficientnetv2_m', num_classes=102, cbam_layers=None):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.cbam_layers = cbam_layers or []
        self.cbam_modules = nn.ModuleDict()

        for name, module in self.backbone.named_modules():
            if any(name.endswith(key) for key in self.cbam_layers):
                if isinstance(module, nn.Conv2d):
                    self.cbam_modules[name.replace('.', '_')] = CBAM(module.out_channels)

    def forward(self, x):
        # Extract features before classifier
        x = self.backbone.forward_features(x)

        # Skip CBAM during feature extraction to avoid dimension mismatch
        # The trained model may have CBAM at different layers than expected

        return x

# ✅ For CBAM

# If you want baseline:
# import timm
# If you want ECA:
# from efficientnetv2_eca import EfficientNetV2WithECA

class IP102Dataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        self.samples = [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)


def extract_features_for_dataset(config, dataset_type='train'):
    """Extract features for train, val, test, or combined train+val dataset"""

    # Choose annotation file(s) based on dataset type
    if dataset_type == 'train':
        annotation_files = [config['train_annotation']]
    elif dataset_type == 'val':
        annotation_files = [config['val_annotation']]
    elif dataset_type == 'test':
        annotation_files = [config['test_annotation']]
    elif dataset_type == 'train_val':
        # Combine train and validation sets
        annotation_files = [config['train_annotation'], config['val_annotation']]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # Create combined dataset from multiple annotation files
    all_samples = []
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        samples = [line.strip().split() for line in lines]
        all_samples.extend(samples)

    # Create custom dataset class for combined samples
    class CombinedIP102Dataset(Dataset):
        def __init__(self, samples, image_dir, transform=None):
            self.samples = samples
            self.image_dir = image_dir
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, int(label)

    dataset = CombinedIP102Dataset(all_samples, config['image_dir'], transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    #print(f"Total samples in {dataset_type} dataset: {len(dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = EfficientNetV2WithCBAM(
    #     model_name=config['model_name'],
    #     num_classes=0,
    #     cbam_layers=config['cbam_layers']
    # ).to(device)

    model = timm.create_model(
        config['model_name'], pretrained=True, num_classes=102
    ).to(device)

    # CBam Load trained weights
    #trained_path = os.path.join(config['log_dir'], "aug_m_cbam_best_model.pth")

    # Original Ef2
    trained_path = os.path.join(config['log_dir'], "aug_baseline_m_best_model.pth")

    state_dict = torch.load(trained_path, map_location=device)
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(filtered_state)
    model.load_state_dict(model_state)

    model.eval()

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)
            feats = model(img)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
            img_path, _ = dataset.samples[i]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(config['log_dir'],"m_baseline_features", f"{img_name}.npy")
            np.save(save_path, feats.cpu().numpy())


    print(f"✅ Done extracting features for {dataset_type} dataset.")


if __name__ == "__main__":
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Extract features from both datasets
    extract_features_for_dataset(config, 'train_val')
    extract_features_for_dataset(config, 'test')



