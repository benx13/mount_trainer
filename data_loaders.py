import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

def create_data_loaders(
    data_dir,
    input_shape, 
    batch_size,
    val_split=0.1,
    test_split=0.1,
    num_workers=None,
    seed=42
):
    """
    Creates data loaders for training, validation, and testing using data from disk in ImageNet format.
    Expects data_dir to have subdirectories 'human' and 'no-human'.

    Args:
        data_dir (str): Directory containing the dataset in ImageNet format
        input_shape (tuple): The desired input shape for images (height, width, channels)
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation (default: 0.1)
        test_split (float): Fraction of data to use for testing (default: 0.1)
        num_workers (int, optional): Number of workers for data loading. If None, uses CPU count
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader) - PyTorch DataLoaders for each split
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Data Preprocessing and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_shape[0], input_shape[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_shape[0], input_shape[1])),
            transforms.ToTensor(),
        ])
    }

    # Load the full dataset
    full_dataset = ImageFolder(data_dir, transform=data_transforms['val'])
    
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
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    # Calculate appropriate number of workers if not specified
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)  # Use at most 8 workers

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

    print(f"\nDataset splits:")
    print(f"Total images: {total_size}")
    print(f"Training: {train_size} images")
    print(f"Validation: {val_size} images")
    print(f"Test: {test_size} images")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to idx mapping: {full_dataset.class_to_idx}\n")

    return train_loader, val_loader, test_loader