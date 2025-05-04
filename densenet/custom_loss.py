import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """
    Combined loss function that includes CrossEntropyLoss and Dice Loss.
    The combined loss is a weighted sum of the two losses.
    Args:
        alpha (float): Weight for CrossEntropyLoss. The weight for Dice Loss is (1 - alpha).
        smooth (float): Smoothing factor for Dice Loss to avoid division by zero.
    """
    def __init__(self, alpha=0.25, smooth=1e-8):  # alpha balances the two losses
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        loss_ce = self.ce(preds, targets)
        loss_dice = 1-self.dice(preds, targets)
        return self.alpha * loss_ce + (1 - self.alpha) * loss_dice

    def dice_per_class(self, preds, targets):
        """
        This function computes the Dice score for each slide. And outputs
        the average Dice score for all slides.
        Args:
            preds (torch.Tensor): The predicted mask of shape (B, H, W).
            targets (torch.Tensor): The ground truth mask of shape (B, H, W).
        Returns:
            float: The average Dice score for all slides.
        """
        B, H, W = targets.shape
        total_dice = 0
        for i in range(B):
            intersection = torch.sum(preds[i] * targets[i])
            union = torch.sum(preds[i]) + torch.sum(targets[i])
            dice = (2*intersection + self.smooth)/(union + self.smooth) 
            total_dice += dice
        return total_dice/B
        

    def dice(self, preds, targets):
        """
        This function computes the Dice score for each class. And outputs
        the average Dice score for all classes.
        Args:
            preds (torch.Tensor): The predicted mask of shape (B, C, H, W).
            targets (torch.Tensor): The ground truth mask of shape (B, C, H, W).
        Returns:
            float: The average Dice score for all classes.
        """

        B, C, H, W = targets.shape
        dice_for_each_class = 0
        
        for i in range(C):
            dice_for_each_class += self.dice_per_class(preds[:,i,:,:], targets[:,i,:,:])
        return dice_for_each_class/C
        
