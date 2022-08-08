from enum import Enum
from torch import Tensor, nn
import torch
from torch.nn import functional as F

def get_loss(type:str):
    if type=='bce':
        return nn.MultiLabelSoftMarginLoss()
    elif type=='partial_bce':
        # parameters selected according to *Learning a Deep ConvNet for Multi-Label Classification With Partial Labels*
        return PartialBCE(-4.45,5.45,1.)
    else:
        raise Exception(f"unknown loss `{type}`. `bce` and `partial_bce` is supported.")



class PartialBCE(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs: Tensor, targets: Tensor, weights=None):
        masks=targets.detach().clone()
        # generate mask. only 1 and -1 are valid labels.
        masks[masks==-1]=1
        # set -1 as 0 to fit standard BCE loss
        targets[targets==-1]=0

        batch_size, num_class = outputs.size()
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # BCEloss needs targets be float
        loss = criterion(outputs, targets.float())
        if weights is not None:
            loss = loss * weights
        # masks==0 are masked
        loss = loss * masks.float()
        known_ys = masks.float().sum(1)
        p_y = known_ys / num_class
        g_p_y = self.alpha * (p_y**self.gamma) + self.beta
        print(g_p_y,num_class,loss.sum(1))
        loss = ((g_p_y / num_class) * loss.sum(1)).mean()
        return loss
