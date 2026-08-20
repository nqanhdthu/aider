import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import yaml
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

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



def extract_and_save_features(dataset_type='train'):
    """Extract features for each image and save separately"""
    # Load config
    with open("train_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    model_path = 'logs/efficientnetv2_eca/aug_baseline_m_best_model.pth'
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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(
        config['model_name'],
        pretrained=False,
        num_classes=config['num_classes']
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Remove classifier to get features
    model.reset_classifier(0)

    input_key = config["input_key"]
    input_size = config['input_size'][input_key]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

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


    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(loader, desc=f"Extracting {dataset_type}")):
            img = img.to(device)
            feats = model(img)
            img_path, _ = dataset.samples[i]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(config['log_dir'],"m_baseline_features", f"{img_name}.npy")
            np.save(save_path, feats.cpu().numpy())

if __name__ == "__main__":
    # Extract features for all datasets
    extract_and_save_features('train_val')
    extract_and_save_features('test')
