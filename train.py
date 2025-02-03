import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders import create_data_loaders
from mcunet.model_zoo import get_specialized_network
import argparse
from trainer import Trainer

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = get_specialized_network(
        task='classification',
        dataset='imagenet',
        model_name='mcunet-320kb',
        pretrained=True,
        num_classes=2  # binary classification: human vs no-human
    )
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

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs
    )

    # Load checkpoint if provided
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed training from checkpoint: {args.resume_from}")

    # Train the model
    trainer.train()

    # Test the model
    test_loss, test_acc = trainer.test()
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCUNet model on Wake Vision dataset")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the dataset in ImageNet format (human/no-human subdirs)")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                      help="Fraction of data to use for testing")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512,
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