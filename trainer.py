import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb
from torch.amp import GradScaler, autocast # Import GradScaler and autocast

def save_checkpoint(state, is_best, output_dir, model_name):
    """Save model checkpoint and optionally log to wandb."""
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)
        wandb.save(best_path)
        return best_path
    return latest_path

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
    resume_training=False
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

    model = model.to(memory_format=torch.channels_last)
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Disable progress bar for faster training
        train_loop = tqdm(train_loader, leave=False, disable=True)
        for images, labels in train_loop:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        train_accuracy = correct_predictions / total_samples
        train_loss = running_loss / len(train_loader)
        
        wandb.log({
            'train/loss': train_loss,
            'train/accuracy': train_accuracy,
            'train/epoch': epoch + 1
        })

        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        # Disable progress bar for validation too
        val_loop = tqdm(val_loader, leave=False, disable=True)
        with torch.no_grad():
            for images, labels in val_loop:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss = val_loss / len(val_loader)

        wandb.log({
            'val/loss': val_loss,
            'val/accuracy': val_accuracy,
            'val/epoch': epoch + 1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"LR: {current_lr:.2e}")

        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(val_accuracy, best_val_accuracy)
        if is_best:
            wandb.run.summary['best_val_accuracy'] = best_val_accuracy

        checkpoint = {
            'epoch': epoch + 1,
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

    from mcunet.model_zoo import build_model
    best_model = build_model(net_id=net_id, pretrained=False)[0]
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device, memory_format=torch.channels_last)
    best_model.eval()

    test_correct = 0
    total = 0
    test_loop = tqdm(test_loader, leave=False, disable=True)
    with torch.no_grad():
        for images, labels in test_loop:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / total
    
    wandb.run.summary.update({
        'test/accuracy': test_accuracy,
        'test/correct': test_correct,
        'test/total': total,
        'test/wrong': total - test_correct
    })
    
    print(f"\nTest accuracy of the best model: {test_accuracy:.4f}\n")

    wandb.finish()

    return best_val_accuracy, test_accuracy
