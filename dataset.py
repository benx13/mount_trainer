import os
import cv2
import torch
from torch.utils.data import Dataset

class AlbumentationsDataset(Dataset):
    """Custom dataset that uses Albumentations for augmentation"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, filename),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Read image using cv2
        image = cv2.imread(img_path)
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
