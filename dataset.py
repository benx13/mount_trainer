import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class AlbumentationsDataset(Dataset):
    """Custom dataset that uses Albumentations for augmentation"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Preload all file paths at init
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            # Use list comprehension for faster file collection
            self.samples.extend([
                (os.path.join(class_dir, filename), class_idx)
                for filename in os.listdir(class_dir)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Use IMREAD_COLOR flag for faster reading and disable auto-orientation
        image = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            # Fallback to PIL if OpenCV fails
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label

class TransformWrapper(Dataset):
    """Wrapper dataset that applies Albumentations transforms to an existing dataset"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)  # Convert to numpy for Albumentations
        transformed = self.transform(image=image)
        return transformed["image"], label
    
    @property
    def classes(self):
        """Pass through dataset classes if available"""
        if hasattr(self.dataset, 'classes'):
            return self.dataset.classes
        elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'classes'):
            return self.dataset.dataset.classes
        return None
