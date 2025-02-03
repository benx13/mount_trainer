import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from datasets import AlbumentationsDataset, TransformWrapper

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

    # Define augmentations using Albumentations
    train_transform = A.Compose([
        A.RandomResizedCrop(height=input_shape[0], width=input_shape[1], scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

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