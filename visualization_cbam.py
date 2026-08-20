import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, class_idx=None):
        model_output = self.model(input_image)

        if class_idx is None:
            class_idx = np.argmax(model_output.cpu().data.numpy())

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(model_output)
        one_hot[0][class_idx] = 1.0
        model_output.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[-1], input_image.shape[-2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam, class_idx


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


class EfficientNetV2WithCBAMForGradCAM(nn.Module):
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
        # Keep the classifier for GradCAM
        return self.backbone(x)

    def get_target_layer(self):
        # Return the last convolutional layer before global pooling
        for name, module in self.backbone.named_modules():
            if 'blocks' in name and isinstance(module, nn.Conv2d):
                target_layer = module
        return target_layer


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
        img_path = os.path.join(self.image_dir, path)
        img = Image.open(img_path).convert('RGB')
        original_img = np.array(img)

        if self.transform:
            img = self.transform(img)
        return img, int(label), original_img, path


def visualize_gradcam_features():
    # Load config
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset (using test set for visualization)
    dataset = IP102Dataset(config['test_annotation'], config['image_dir'], transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with classifier for GradCAM
    model = EfficientNetV2WithCBAMForGradCAM(
        model_name=config['model_name'],
        num_classes=102,
        cbam_layers=config['cbam_layers']
    ).to(device)

    # Load trained weights
    trained_path = os.path.join(config['log_dir'], "aug_m_cbam_best_model.pth")
    state_dict = torch.load(trained_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get target layer for GradCAM
    target_layer = model.get_target_layer()

    # Create GradCAM instance
    grad_cam = GradCAM(model, target_layer)

    # Create output directory
    output_dir = os.path.join(config['log_dir'], "gradcam_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Visualize first 5 samples
    num_samples = 5
    samples_processed = 0

    print(f"Generating GradCAM visualizations...")

    for imgs, labels, original_imgs, paths in tqdm(loader, desc="Processing"):
        if samples_processed >= num_samples:
            break

        img = imgs.to(device)
        label = labels.item()
        original_img = original_imgs[0]  # This is already a numpy array
        img_path = paths[0]

        # Ensure original_img is numpy array and convert to uint8
        if not isinstance(original_img, np.ndarray):
            original_img = np.array(original_img)

        # Ensure proper data type for cv2
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)

        # Generate GradCAM
        with torch.no_grad():
            # Get model prediction
            output = model(img)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

        # Generate CAM for predicted class
        cam, _ = grad_cam.generate_cam(img, predicted_class)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Image\nTrue Label: {label}')
        axes[0].axis('off')

        # GradCAM heatmap
        axes[1].imshow(cam, cmap='jet', alpha=0.8)
        axes[1].set_title(f'GradCAM Heatmap\nPred: {predicted_class} ({confidence:.3f})')
        axes[1].axis('off')

        # Overlay - Fix the cv2.resize issue
        resized_original = cv2.resize(original_img, (input_size, input_size))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = heatmap * 0.4 + resized_original * 0.6
        overlay = overlay.astype(np.uint8)

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nAttention Regions')
        axes[2].axis('off')

        plt.tight_layout()

        # Save visualization
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"gradcam_{samples_processed}_{img_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Also save individual components
        # Save original
        plt.figure(figsize=(8, 8))
        plt.imshow(original_img)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"original_{samples_processed}_{img_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"heatmap_{samples_processed}_{img_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"overlay_{samples_processed}_{img_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        samples_processed += 1

        print(f"Processed sample {samples_processed}: {img_name}")
        print(f"  True label: {label}, Predicted: {predicted_class}, Confidence: {confidence:.3f}")

    print(f"✅ GradCAM visualizations saved to {output_dir}")
    print(f"Generated visualizations for {samples_processed} samples")


def visualize_class_specific_gradcam(target_classes=None):
    """Visualize GradCAM for specific classes"""
    # Load config
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    if target_classes is None:
        target_classes = [0, 1, 2, 3, 4]  # First 5 classes

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IP102Dataset(config['test_annotation'], config['image_dir'], transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EfficientNetV2WithCBAMForGradCAM(
        model_name=config['model_name'],
        num_classes=102,
        cbam_layers=config['cbam_layers']
    ).to(device)

    trained_path = os.path.join(config['log_dir'], "aug_m_cbam_best_model.pth")
    state_dict = torch.load(trained_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = model.get_target_layer()
    grad_cam = GradCAM(model, target_layer)

    output_dir = os.path.join(config['log_dir'], "class_specific_gradcam")
    os.makedirs(output_dir, exist_ok=True)

    for target_class in target_classes:
        print(f"Searching for samples of class {target_class}...")

        class_samples = []
        for i, (img, label, original_img, path) in enumerate(dataset):
            if label == target_class:
                class_samples.append((img, label, original_img, path))
                if len(class_samples) >= 3:  # Get 3 samples per class
                    break

        if len(class_samples) == 0:
            print(f"No samples found for class {target_class}")
            continue

        for idx, (img, label, original_img, path) in enumerate(class_samples):
            img_tensor = img.unsqueeze(0).to(device)

            # Generate GradCAM for true class
            cam, _ = grad_cam.generate_cam(img_tensor, target_class)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_img)
            axes[0].set_title(f'Class {target_class} - Sample {idx + 1}')
            axes[0].axis('off')

            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('GradCAM Heatmap')
            axes[1].axis('off')

            resized_original = cv2.resize(original_img, (input_size, input_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = heatmap * 0.4 + resized_original * 0.6
            overlay = overlay.astype(np.uint8)

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()

            img_name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(output_dir, f"class_{target_class}_sample_{idx}_{img_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Generated {len(class_samples)} visualizations for class {target_class}")

    print(f"✅ Class-specific GradCAM visualizations saved to {output_dir}")


if __name__ == "__main__":
    #Generate general GradCAM visualizations
    #visualize_gradcam_features()

    # Generate class-specific visualizations
    visualize_class_specific_gradcam(target_classes=[0, 1, 2, 3, 4, 10, 15, 20])