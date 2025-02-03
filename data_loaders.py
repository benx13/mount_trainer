import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def create_data_loaders(input_shape, batch_size, use_relabeled_data=False, relabeled_dataset_csv='relabeled_dataset.csv'):
    """
    Creates data loaders for training, validation, and testing with preprocessing and augmentation.

    Args:
        input_shape (tuple): The desired input shape for images (height, width, channels).
        batch_size (int): Batch size for data loaders.
        use_relabeled_data (bool, optional): Whether to use relabeled data. Defaults to False.
        relabeled_dataset_csv (str, optional): Path to the CSV file with relabeled data. Defaults to 'relabeled_dataset.csv'.

    Returns:
        tuple: (train_loader, val_loader, test_loader) - PyTorch DataLoaders for each split.
    """

    # Load dataset - same as Keras script
    ds = load_dataset("Harvard-Edge/Wake-Vision")
    train_ds = ds['train_quality']
    val_ds = ds['validation']
    test_ds = ds['test']

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn) # Batch size 1 for test as in Keras script

    return train_loader, val_loader, test_loader