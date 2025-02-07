import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb
from torch.amp import GradScaler, autocast # Import GradScaler and autocast
import torch.distributed as dist
import math

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

def get_scaled_lr(base_lr, batch_size, world_size):
    """
    Scale learning rate based on global batch size.
    Uses the linear scaling rule: lr = base_lr * (global_batch_size / 256)
    
    Args:
        base_lr: Base learning rate for batch size 256
        batch_size: Batch size per GPU
        world_size: Number of GPUs
    """
    global_batch_size = batch_size * world_size
    scale_factor = global_batch_size / 256
    return base_lr * scale_factor

def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Create a learning rate scheduler with linear warmup and cosine decay
    """
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    rank=0,
    world_size=1,
    warmup_epochs=5,  # Add warmup epochs parameter
    base_lr=0.001     # Add base learning rate parameter
):
    """
    Train a model with support for distributed training.
    
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
        rank: Rank of the current process
        world_size: Total number of processes
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate
    
    Returns:
        float: Best validation accuracy achieved
        float: Test accuracy of the best model
    """
    # Scale learning rate based on global batch size
    batch_size = train_loader.batch_size
    scaled_lr = get_scaled_lr(base_lr, batch_size, world_size)
    

    
    # Adjust optimizer with scaled learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = scaled_lr
    
    # Create warmup scheduler
    steps_per_epoch = len(train_loader)
    warmup_scheduler = get_warmup_scheduler(
        optimizer, 
        warmup_epochs, 
        epochs, 
        steps_per_epoch
    )
    
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
        # Adjust warmup scheduler to resume point
        for _ in range(start_epoch * steps_per_epoch):
            warmup_scheduler.step()
    else:
        start_epoch = 0
        best_val_accuracy = 0.0

    for epoch in range(start_epoch, epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Set train mode
        model.train()
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        
        # Create progress bar for training
        if rank == 0:  # Only show progress bar on main process
            train_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        else:
            train_iter = train_loader
            
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total_samples += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar on main process
            if rank == 0:
                train_iter.set_postfix({
                    'loss': train_loss / (train_iter.n + 1e-5),
                    'acc': train_correct_predictions / train_total_samples
                })

        # Calculate epoch metrics
        train_accuracy = train_correct_predictions / train_total_samples
        train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        
        # Create progress bar for validation
        if rank == 0:
            val_iter = tqdm(val_loader, desc=f"Validation", leave=False)
        else:
            val_iter = val_loader
            
        with torch.no_grad():
            for images, labels in val_iter:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar on main process
                if rank == 0:
                    val_iter.set_postfix({
                        'loss': val_loss / (val_iter.n + 1e-5),
                        'acc': val_correct_predictions / val_total_samples
                    })

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss = val_loss / len(val_loader)

        # Log validation metrics to wandb
        if rank == 0:
            wandb.log({
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'val/epoch': epoch + 1,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Step both schedulers
        warmup_scheduler.step()
        
        # Only step the main scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        # Log learning rate
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'learning_rate': current_lr,
                'epoch': epoch
            })
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

        # Save checkpoint
        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(val_accuracy, best_val_accuracy)
        
        # Log best metrics to wandb
        if is_best and rank == 0:
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
    if rank == 0:
        wandb.run.summary.update({
            'test/accuracy': test_accuracy,
            'test/correct': test_correct,
            'test/total': total,
            'test/wrong': total - test_correct
        })
    
    print(f"\nTest accuracy of the best model: {test_accuracy:.4f}\n")

    # Only rank 0 should finish the wandb run
    if rank == 0:
        wandb.finish()

    return best_val_accuracy, test_accuracy
