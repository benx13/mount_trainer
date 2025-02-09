import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from data_loaders import create_data_loaders
from mcunet.model_zoo import build_model
import argparse
from trainer import train_model
from lebel_smooth import LabelSmoothingLoss
from torch.amp import GradScaler, autocast

def main(args):
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "net_id": args.net_id,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "seed": args.seed
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})

    # Enable cuDNN benchmarking and deterministic mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Create model using build_model
    model, image_size, description = build_model(
        net_id=args.net_id,
        pretrained=True,
    )
    print(f"Loaded model: {args.net_id}")
    print(f"Image size: {image_size}")
    print(f"Description: {description}")
    
    # Use channels_last memory format for better performance on NVIDIA GPUs
    model = model.to(device, memory_format=torch.channels_last)

    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        model = torch.compile(model)  # PyTorch 2.0+ optimization

    # Enable gradient checkpointing if model supports it
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Optimize batch size and workers based on GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        #suggested_batch_size = min(args.batch_size, int(gpu_mem * 4))  # Rough estimate
        #suggested_workers = min(os.cpu_count(), int(gpu_mem * 2))
        
        #args.batch_size = 512#suggested_batch_size
        #args.num_workers = suggested_workers
        
        print(f"Optimized batch size: {args.batch_size}")
        print(f"Optimized num workers: {args.num_workers}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        input_shape=(144, 144, 3),  # Model's expected input size
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        # Add these parameters for faster data loading
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Define loss function, optimizer and scheduler
    #criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLoss(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    scaler = GradScaler('cuda')  # Initialize GradScaler

    # Train the model using train_model function
    best_val_accuracy, test_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
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
    
    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="mcunet-training",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (username or team name)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Weights & Biases run name")
    
    # Model arguments
    parser.add_argument("--net_id", type=str, default="mcunet-vww2",
                      help="Model ID from MCUNet model zoo")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the training dataset in ImageNet format (human/no-human subdirs)")
    parser.add_argument("--val_dir", type=str,
                      help="Optional directory containing validation dataset. If provided, val_split is ignored")
    parser.add_argument("--test_dir", type=str,
                      help="Optional directory containing test dataset. If provided, test_split is ignored")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation when val_dir is not provided")
    parser.add_argument("--test_split", type=float, default=0.1,
                      help="Fraction of data to use for testing when test_dir is not provided")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=16,
                      help="Number of workers for training")
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
    
    # Validate directory arguments
    if (args.val_dir and not args.test_dir) or (args.test_dir and not args.val_dir):
        parser.error("If providing separate validation/test directories, both --val_dir and --test_dir must be specified")
    
    if args.val_dir and args.test_dir:
        print("\nUsing separate directories for validation and test sets")
        for dir_path in [args.val_dir, args.test_dir]:
            if not os.path.exists(dir_path):
                parser.error(f"Directory does not exist: {dir_path}")
    else:
        print("\nUsing splits from training directory for validation and test sets")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    main(args)