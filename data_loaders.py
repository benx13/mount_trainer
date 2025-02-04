import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from datasets import AlbumentationsDataset, TransformWrapper

def get_augmentation_pipeline(train=True, img_size=224):
    if train:
        augmentation_list = [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=0.1, max_width=0.1, min_holes=4, min_height=0.05, min_width=0.05, p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.02, 0.02), rotate=(-5, 5), shear=(-2, 2), p=0.2),
            A.RandomRain(brightness_coefficient=0.9, drop_length=10, drop_width=1, blur_value=7, rain_type='drizzle', p=0.1),
            A.RandomFog(fog_limit=(10, 30), alpha_coef=0.08, p=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.1),
            A.ISONoise(intensity=(0.1, 0.3), color_shift=(0.01, 0.03), p=0.1),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.1),
            A.ElasticTransform(alpha=1, sigma=25, alpha_affine=25, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.05),
            A.GridDistortion(num_steps=5, distort_limit=(-0.1, 0.1), interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.05),
            A.ToGray(p=0.05),
            A.FancyPCA(alpha=0.05, p=0.05),
            A.MixUp(alpha=0.2, p=0.02),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=(0,0,0), p=0.2),
            A.CutMix(num_mix=2, p=0.02),
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        augmentation_list = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    return A.Compose(augmentation_list, p=1.0, additional_targets={'mask': 'mask'})

def create_data_loaders(
    data_dir,
    input_shape, 
    batch_size,
    val_split=0.1,
    test_split=0.1,
    num_workers=None,
    seed=42,
    val_dir=None,
    test_dir=None
):
    """
    Creates data loaders for training, validation, and testing using data from disk in ImageNet format.
    If val_dir and test_dir are provided, uses those directories for validation and test sets.
    Otherwise, splits data_dir into train/val/test sets using the specified splits.
    Uses Albumentations for data augmentation.

    Args:
        data_dir (str): Directory containing the training dataset in ImageNet format
        input_shape (tuple): The desired input shape for images (height, width, channels)
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation when splitting (default: 0.1)
        test_split (float): Fraction of data to use for testing when splitting (default: 0.1)
        num_workers (int, optional): Number of workers for data loading. If None, uses CPU count
        seed (int): Random seed for reproducibility
        val_dir (str, optional): Directory containing validation dataset. If provided, val_split is ignored
        test_dir (str, optional): Directory containing test dataset. If provided, test_split is ignored

    Returns:
        tuple: (train_loader, val_loader, test_loader) - PyTorch DataLoaders for each split
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    if val_dir and not os.path.exists(val_dir):
        raise ValueError(f"Validation directory {val_dir} does not exist")
    if test_dir and not os.path.exists(test_dir):
        raise ValueError(f"Test directory {test_dir} does not exist")

    # Get augmentation pipelines
    train_transform = get_augmentation_pipeline(train=True, img_size=input_shape[0])
    val_transform = get_augmentation_pipeline(train=False, img_size=input_shape[0])

    # Calculate appropriate number of workers if not specified
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)  # Use at most 8 workers

    # Case 1: Using separate directories for validation and test
    if val_dir and test_dir:
        train_dataset = AlbumentationsDataset(data_dir, transform=train_transform)
        val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)
        test_dataset = AlbumentationsDataset(test_dir, transform=val_transform)
        
        print(f"\nUsing separate directories for validation and test sets:")
        print(f"Training: {len(train_dataset)} images from {data_dir}")
        print(f"Validation: {len(val_dataset)} images from {val_dir}")
        print(f"Test: {len(test_dataset)} images from {test_dir}")
    
    # Case 2: Split single directory into train/val/test
    else:
        # Load the full dataset
        full_dataset = AlbumentationsDataset(data_dir, transform=val_transform)
        
        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        test_size = int(test_split * total_size)
        train_size = total_size - val_size - test_size

        # Split the dataset
        torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size, test_size]
        )

        # Apply transforms to each split
        train_dataset = TransformWrapper(train_dataset, train_transform)
        val_dataset = TransformWrapper(val_dataset, val_transform)
        test_dataset = TransformWrapper(test_dataset, val_transform)
        
        print(f"\nSplit single directory into train/val/test:")
        print(f"Total images: {total_size} from {data_dir}")
        print(f"Training: {train_size} images")
        print(f"Validation: {val_size} images")
        print(f"Test: {test_size} images")

    # Verify class consistency across splits
    if val_dir and test_dir:
        train_classes = set(train_dataset.classes)
        val_classes = set(val_dataset.classes)
        test_classes = set(test_dataset.classes)
    else:
        train_classes = set(full_dataset.classes)
        val_classes = train_classes
        test_classes = train_classes
    
    if not (train_classes == val_classes == test_classes):
        raise ValueError("Classes are not consistent across train, validation, and test sets")

    print(f"Classes: {sorted(train_classes)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=max(1, num_workers//2),  # Use fewer workers for test set
        pin_memory=True
    )

    return train_loader, val_loader, test_loader