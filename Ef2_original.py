import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
import time

class IP102Dataset(Dataset):
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

if __name__ == "__main__":
    with open("train_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], 'baseline'))

    early_stop_patience = config.get('early_stop_patience', 5)
    early_stop_counter = 0
    input_size = 300 if 's' in config['model_name'] else 384
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

    train_dataset = IP102Dataset(config['train_annotation'], config['image_dir'], transform=train_transform)
    val_dataset = IP102Dataset(config['val_annotation'], config['image_dir'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = timm.create_model(
        config['model_name'], pretrained=True, num_classes=102
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=1e-4)

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
            torch.save(model.state_dict(), os.path.join(config['log_dir'], 'aug_baseline_m_best_model.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    writer.close()
