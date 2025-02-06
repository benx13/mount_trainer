import numpy as np
import torch

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch, targets):
        """Performs mixup on the input batch and targets.
        
        Args:
            batch: Input batch of images (B, C, H, W)
            targets: Target labels
            
        Returns:
            mixed_batch: Mixup batch
            target_a: First target
            target_b: Second target
            lam: Lambda value for mixing
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size()[0]
        index = torch.randperm(batch_size).to(batch.device)

        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        target_a, target_b = targets, targets[index]
        return mixed_batch, target_a, target_b, lam

    def mix_criterion(self, criterion, pred, target_a, target_b, lam):
        """Combines the loss for mixed targets.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            target_a: First target
            target_b: Second target
            lam: Lambda value used for mixing
            
        Returns:
            Combined loss value
        """
        return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
