import torch
import torch.nn as nn
import torch.nn.functional as F

class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss
    https://arxiv.org/abs/1908.06112
    """
    def __init__(self, alpha=1.0, beta=1.0, num_classes=2, smoothing=0.0):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.cross_entropy = nn.CrossEntropyLoss()
        self.register_buffer('uniform_labels', torch.ones(num_classes) / num_classes)
        self.register_buffer('eps', torch.tensor(1e-7))
        self.register_buffer('one_hot_eps', torch.tensor(1e-4))

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE - use more efficient operations
        with torch.amp.autocast('cuda', enabled=False):  # Use fp32 for stability
            pred_softmax = F.softmax(pred.float(), dim=1).clamp(min=self.eps)
            # One-hot with smoothing in one operation
            label_one_hot = F.one_hot(labels, self.num_classes).float()
            if self.smoothing > 0.0:
                label_one_hot = label_one_hot * (1 - self.smoothing) + self.smoothing * self.uniform_labels
            label_one_hot = label_one_hot.clamp(min=self.one_hot_eps)
            
            # Compute RCE more efficiently
            rce = -(pred_softmax * label_one_hot.log()).sum(dim=1)

        # Combine losses
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss 