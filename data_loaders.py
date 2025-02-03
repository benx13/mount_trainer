import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from download import load_wake_vision_dataset

def create_data_loaders(
    input_shape, 
    batch_size, 
    use_relabeled_data=False, 
    relabeled_dataset_csv='relabeled_dataset.csv',
    num_proc=8,
    num_shards=1,
    shard_id=0
):
    """
    Creates data loaders for training, validation, and testing with preprocessing and augmentation.

    Args:
        input_shape (tuple): The desired input shape for images (height, width, channels).
        batch_size (int): Batch size for data loaders.
        use_relabeled_data (bool, optional): Whether to use relabeled data. Defaults to False.
        relabeled_dataset_csv (str, optional): Path to the CSV file with relabeled data. Defaults to 'relabeled_dataset.csv'.
        num_proc (int, optional): Number of processes for data loading. Defaults to 8.
        num_shards (int, optional): Total number of shards to divide the dataset into. Defaults to 1.
        shard_id (int, optional): Shard ID to load (0-indexed). Defaults to 0.

    Returns:
        tuple: (train_loader, val_loader, test_loader) - PyTorch DataLoaders for each split.
    """

    # Load dataset using optimized loader from download.py
    train_ds = load_wake_vision_dataset(
        dataset_name="Harvard-Edge/Wake-Vision",
        split='train_quality',
        streaming=False,
        num_proc=num_proc,
        num_shards=num_shards,
        shard_id=shard_id
    )
    
    val_ds = load_wake_vision_dataset(
        dataset_name="Harvard-Edge/Wake-Vision",
        split='validation',
        streaming=False,
        num_proc=num_proc,
        num_shards=1,  # Don't shard validation set
        shard_id=0
    )
    
    test_ds = load_wake_vision_dataset(
        dataset_name="Harvard-Edge/Wake-Vision",
        split='test',
        streaming=False,
        num_proc=num_proc,
        num_shards=1,  # Don't shard test set
        shard_id=0
    )

    # Data Preprocessing and Augmentation - PyTorch/Torchvision style
    data_preprocessing = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])), # Resize images
        transforms.ToTensor(), # Convert PIL Image to Tensor and normalize to [0, 1]
    ])

    data_augmentation = transforms.Compose([
        data_preprocessing,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20), # Slightly different rotation angle, adjust if needed to match Keras
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Optional: ImageNet normalization if pretrained model expects it
    ])

    def transform_train(examples):
        examples["image"] = [data_augmentation(image.convert("RGB")) for image in examples["image"]] # Convert to RGB
        return examples

    def transform_val_test(examples):
        examples["image"] = [data_preprocessing(image.convert("RGB")) for image in examples["image"]] # Convert to RGB for val/test, no augmentation
        return examples

    train_ds.set_transform(transform_train)
    val_ds.set_transform(transform_val_test)
    test_ds.set_transform(transform_val_test)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        # --- MODIFIED LABEL LOADING ---
        if use_relabeled_data:
            # Assuming you have a way to map image filenames in batch to labels from relabeled_dataset_csv
            # You'll need to implement the logic to load labels from your CSV based on filenames in the batch
            # This is a placeholder - you'll need to adapt it to your CSV structure and loading method
            print("Warning: Relabeled data usage is not fully implemented in collate_fn. You need to load labels from CSV here.")
            labels = torch.tensor([item['person'] for item in batch], dtype=torch.long) # Placeholder: Replace with actual relabeled labels
        else:
            labels = torch.tensor([item['person'] for item in batch], dtype=torch.long) # Use original 'person' label
        # --- END MODIFIED LABEL LOADING ---
        return images, labels

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_proc,  # Match num_workers with num_proc
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_proc,  # Match num_workers with num_proc
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=1,  # Batch size 1 for test as in Keras script
        shuffle=False, 
        num_workers=num_proc//2,  # Use fewer workers for test set
        pin_memory=True, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader