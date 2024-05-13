from torch import nn
import torch


class BolTLoss(nn.Module):
    def __init__(self, lambda_cons: float) -> torch.Tensor:
        """Custom loss for BolT architecture which penalizes
        the deviation of individual CLS tokens from their mean over windows
        """

        super(BolTLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_cons = lambda_cons

    def forward(self, output, y):
        logits, cls = output
        cls_loss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))
        cross_entropy_loss = self.criterion(logits, y)
        return cross_entropy_loss + cls_loss * self.lambda_cons
