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