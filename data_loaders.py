import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from dataset import AlbumentationsDataset, TransformWrapper, CachedImageDataset
from torch.utils.data.distributed import DistributedSampler

def train_augmentation_pipeline(img_size: int):
    return A.Compose([
        # 1. Always crop & resize to target dimensions
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),

        # 2. Basic horizontal flip
        A.HorizontalFlip(p=0.5),

        # 3. Light color transformations (choose 1 out of the 4)
        A.SomeOf(
            transforms=[
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.ToGray(p=1.0),
                A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0)
            ],
            n=1,              # Pick exactly 1 transform to apply
            p=0.4             # 40% chance to apply any color augmentation
        ),

        # 4. Apply a blur or motion blur with moderate kernel sizes
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),

        # 5. Mild geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.05,     # small shift
            scale_limit=0.1,      # up to +/-10% scale
            rotate_limit=10,      # up to +/-10 degrees
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.2
        ),

        # 6. Occasional random weather condition
        A.OneOf([
            A.RandomRain(
                brightness_coefficient=0.9,
                drop_length=8,
                drop_width=1,
                blur_value=5,
                rain_type='drizzle',
                p=1.0
            ),
            A.RandomFog(fog_limit=(10, 20), alpha_coef=0.05, p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                p=1.0
            ),
        ], p=0.15),

        # 7. Occasional Coarse Dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=4,
            min_height=8,
            min_width=8,
            p=0.2
        ),

        # 8. Normalize & convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def val_augmentation_pipeline(img_size: int):
    """Validation/Test transforms - just resize and normalize"""
    return A.Compose([
        A.Resize(
            height=img_size,
            width=img_size,
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def create_data_loaders(
    data_dir,
    input_shape, 
    batch_size,
    val_split=0.1,
    test_split=0.1,
    num_workers=None,
    seed=42,
    val_dir=None,
    test_dir=None,
    world_size=1,
    rank=-1,
    cache_dir=None,
    persistent_workers=True
):
    """Optimized data loaders for large-scale training"""
    
    # Optimize number of workers per GPU
    if num_workers is None:
        num_workers = 4  # Reduced workers per GPU to prevent I/O bottleneck
    
    # Increase prefetch factor for better GPU utilization
    prefetch_factor = 2  # Reduced to prevent memory pressure
    
    # Pin memory for faster data transfer to GPU
    pin_memory = True
    
    # Get augmentation pipelines
    train_transform = train_augmentation_pipeline(input_shape[0])
    val_transform = val_augmentation_pipeline(input_shape[0])
    
    # Create datasets without caching
    if val_dir and test_dir:
        train_dataset = AlbumentationsDataset(
            data_dir, 
            transform=train_transform, 
            skip_corrupt=True,
            num_workers=num_workers
        )
        val_dataset = AlbumentationsDataset(
            val_dir, 
            transform=val_transform, 
            skip_corrupt=True,
            num_workers=num_workers
        )
        test_dataset = AlbumentationsDataset(
            test_dir, 
            transform=val_transform, 
            skip_corrupt=True,
            num_workers=num_workers
        )
        
        print(f"\nUsing separate directories for validation and test sets:")
        print(f"Training: {len(train_dataset)} images from {data_dir}")
        print(f"Validation: {len(val_dataset)} images from {val_dir}")
        print(f"Test: {len(test_dataset)} images from {test_dir}")
    
    # Case 2: Split single directory into train/val/test
    else:
        # Load the full dataset
        full_dataset = AlbumentationsDataset(
            data_dir, 
            transform=val_transform, 
            skip_corrupt=True,
            num_workers=num_workers
        )
        
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
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=max(1, num_workers//2),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader