from enum import Enum
from torch import Tensor, nn
import torch

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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs: Tensor, targets: Tensor, weights=None):
        masks=targets.clone()
        # generate mask. only 1 and -1 are valid labels.
        masks[masks==-1]=1
        # set -1 as 0 to fit standard BCE loss
        targets[targets==-1]=0

        batch_size, chan = outputs.size()
        criterion = torch.nn.BCELoss(reduction="none").cuda()
        loss = criterion(outputs, targets)
        if weights is not None:
            loss = loss * weights
        # masks==0 are masked
        loss = loss * masks.float()
        known_ys = masks.float().sum(1)
        p_y = known_ys / chan
        g_p_y = self.alpha * (p_y**self.gamma) + self.beta
        loss = ((g_p_y / chan) * loss.sum(1)).mean()
        return loss
