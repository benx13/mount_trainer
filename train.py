import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders import create_data_loaders
from mcunet.model_zoo import build_model
import argparse
from trainer import train_model

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model using build_model
    model, image_size, description = build_model(
        net_id=args.net_id,
        pretrained=True,
        num_classes=2  # binary classification: human vs no-human
    )
    print(f"Loaded model: {args.net_id}")
    print(f"Image size: {image_size}")
    print(f"Description: {description}")
    
    model = model.to(device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        input_shape=(144, 144, 3),  # Model's expected input size
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model using train_model function
    best_val_accuracy, test_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=args.checkpoint_dir,
        model_name=args.net_id,
        net_id=args.net_id,
        num_classes=2,
        resume_training=args.resume_from is not None
    )

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCUNet model on Wake Vision dataset")
    
    # Model arguments
    parser.add_argument("--net_id", type=str, default="mcunet-vww2",
                      help="Model ID from MCUNet model zoo")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the dataset in ImageNet format (human/no-human subdirs)")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                      help="Fraction of data to use for testing")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                      help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str,
                      help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    main(args)