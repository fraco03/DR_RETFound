import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss) for regression tasks.
    This loss is less sensitive to outliers than MSE and combines L1 and L2 behavior.
    """
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()


def label_to_levels(labels, num_classes):
    """
    Converts class labels (0..K-1) to CORAL ordinal levels of length K-1.
    Example: label=2, K=5 -> [1, 1, 0, 0]
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    labels = labels.long()
    levels = torch.zeros((labels.size(0), num_classes - 1), device=labels.device)
    for k in range(num_classes - 1):
        levels[:, k] = (labels > k).float()
    return levels


class CoralLoss(nn.Module):
    """
    CORAL loss implemented as BCEWithLogitsLoss over ordinal levels.
    Inputs: logits [B, K-1], levels [B, K-1]
    """
    def __init__(self, reduction="mean"):
        super(CoralLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, levels):
        return self.bce(logits, levels)