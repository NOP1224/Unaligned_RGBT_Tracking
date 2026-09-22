import torch
import torch.nn as nn
import torch.nn.functional as F

class OffsetLoss(nn.Module):
    def __init__(self, loss_type="L2", beta=1.0):
        """
        Args:
            loss_type: str, "L1" | "L2" | "SmoothL1"
            beta: float, SmoothL1 损失的 beta
        """
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(self, pred, target):
        
        # 选择损失类型
        if self.loss_type == "L1":
            loss = F.l1_loss(pred, target)
        elif self.loss_type == "L2":
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "SmoothL1":
            loss = F.smooth_l1_loss(pred, target, beta=self.beta)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
