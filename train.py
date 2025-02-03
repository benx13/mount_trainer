import torch
import torch.nn as nn
import torch.optim as optim
from mcunet.model_zoo import build_model
import os
import argparse
from data_loaders import create_data_loaders
from trainer import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train MCUNet model')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_name = 'wv_quality_mcunet-vww2' # Using mcunet-vww2 as it's likely the architecture
    net_id = "mcunet-vww2" # net_id for build_model

    # Fixed hyperparameters - keep them consistent
    input_shape = (144, 144, 3) # Keras script uses (144, 144, 3) input shape
    batch_size = 512
    learning_rate = 0.001
    epochs = 100
    num_classes = 2 # Binary classification (person/not person)

    # --- ADDED VARIABLE ---
    use_relabeled_data = False  # Set to True to use relabeled dataset, False to use original labels
    relabeled_dataset_csv = 'relabeled_dataset.csv' # Path to your relabeled CSV file
    # --- END ADDED VARIABLE ---

    # Create data loaders using the function from data_loaders.py
    train_loader, val_loader, test_loader = create_data_loaders(
        input_shape=input_shape,
        batch_size=batch_size,
        use_relabeled_data=use_relabeled_data,
        relabeled_dataset_csv=relabeled_dataset_csv,
        num_proc=8,  # Number of processes for data loading
        num_shards=1,  # Total number of shards to divide the dataset into
        shard_id=0  # Shard ID to load (0-indexed)
    )

    # Load MCUNet model from model_zoo - using mcunet-vww2
    model, image_size, description = build_model(net_id=net_id, pretrained=True, num_classes=num_classes)
    print(f"Loaded model: {net_id}, Image size: {image_size}, Description: {description}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Output directory for checkpoints
    output_dir = "./pytorch_checkpoints"

    # Train the model using the trainer module
    best_val_accuracy, test_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        output_dir=output_dir,
        model_name=model_name,
        net_id=net_id,
        num_classes=num_classes,
        resume_training=args.resume
    )

if __name__ == '__main__':
    main()