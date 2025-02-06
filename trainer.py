import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb
from torch.amp import GradScaler, autocast
from mixup import Mixup  # Import Mixup augmentation

def save_checkpoint(state, is_best, output_dir, model_name):
    """Save model checkpoint and optionally log to wandb."""
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = os.path.join(output_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)
        
        # Log best model to wandb
        wandb.save(best_path)
        
    return latest_path if not is_best else best_path
    """
    Save model checkpoint.
    
    Args:
        state (dict): State dictionary containing model state and metadata
        is_best (bool): Whether this is the best model so far
        output_dir (str): Directory to save checkpoints
        model_name (str): Base name for the checkpoint files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = os.path.join(output_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        scaler: Optional GradScaler to load state
    
    Returns:
        tuple: (start_epoch, best_val_accuracy)
    """
    if not os.path.exists(checkpoint_path):
        return 0, 0.0
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_accuracy', 0.0)

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    output_dir,
    model_name,
    net_id,
    num_classes,
    resume_training=False,
    mixup_alpha=0.2  # Mixup alpha parameter
):
    """
    Train a model with the given parameters and data loaders.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        output_dir: Directory to save checkpoints
        model_name: Name of the model for saving checkpoints
        net_id: ID of the network architecture
        num_classes: Number of output classes
        resume_training: Whether to resume from latest checkpoint
    
    Returns:
        float: Best validation accuracy achieved
        float: Test accuracy of the best model
    """
    # Initialize best accuracy and start epoch
    latest_checkpoint = os.path.join(output_dir, f"{model_name}_latest.pth")
    if resume_training and os.path.exists(latest_checkpoint):
        start_epoch, best_val_accuracy = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )
        print(f"Resuming training from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.4f}")
    else:
        start_epoch = 0
        best_val_accuracy = 0.0

    # Initialize mixup
    mixup = Mixup(alpha=mixup_alpha)
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Create progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

        train_loop = tqdm(train_loader, leave=False)
        for images, labels in train_loop:
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Apply mixup
            mixed_images, target_a, target_b, lam = mixup(images, labels)

            with autocast('cuda'):
                outputs = model(mixed_images)
                loss = mixup.mix_criterion(criterion, outputs, target_a, target_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            train_loop.set_postfix(
                train_loss=running_loss / (train_loop.n + 1e-5),
                train_acc=correct_predictions / total_samples
            )

        train_accuracy = correct_predictions / total_samples
        train_loss = running_loss / len(train_loader)
        
        # Log training metrics to wandb
        wandb.log({
            'train/loss': train_loss,
            'train/accuracy': train_accuracy,
            'train/epoch': epoch + 1
        })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_loop = tqdm(val_loader, leave=False)
        
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                val_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                val_loop.set_postfix(
                    val_loss=val_loss / (val_loop.n + 1e-5),
                    val_acc=val_correct_predictions / val_total_samples
                )

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss = val_loss / len(val_loader)

        # Log validation metrics to wandb
        wandb.log({
            'val/loss': val_loss,
            'val/accuracy': val_accuracy,
            'val/epoch': epoch + 1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"LR: {current_lr:.2e}")

        # Save checkpoint
        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(val_accuracy, best_val_accuracy)
        
        # Log best metrics to wandb
        if is_best:
            wandb.run.summary['best_val_accuracy'] = best_val_accuracy
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch + 1,  # Save next epoch to resume from
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'val_accuracy': val_accuracy,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'train_loss': train_loss
        }
        save_checkpoint(checkpoint, is_best, output_dir, model_name)

    print(f"\nBest validation accuracy achieved: {best_val_accuracy:.4f}")
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    print(f"Best model saved to: {best_model_path}")

    # Testing phase - use the best model for testing
    from mcunet.model_zoo import build_model
    best_model = build_model(net_id=net_id, pretrained=False)[0]
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    best_model.eval()

    test_correct = 0
    total = 0
    test_loop = tqdm(test_loader, leave=False)
    
    with torch.no_grad():
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / total
    
    # Log final test metrics to wandb
    wandb.run.summary.update({
        'test/accuracy': test_accuracy,
        'test/correct': test_correct,
        'test/total': total,
        'test/wrong': total - test_correct
    })
    
    print(f"\nTest accuracy of the best model: {test_accuracy:.4f}\n")

    # Finish the wandb run
    wandb.finish()

    return best_val_accuracy, test_accuracy
