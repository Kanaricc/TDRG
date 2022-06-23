from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

class Baseline(nn.Module):
    def __init__(self,feat_dim:int,num_classes:int) -> None:
        super().__init__()
        res101 = torchvision.models.resnet101(pretrained=True)
        self.layer1 = nn.Sequential(
            res101.conv1,
            res101.bn1,
            res101.relu,
            res101.maxpool,
            res101.layer1,
        )
        self.layer2 = res101.layer2
        self.layer3 = res101.layer3
        self.layer4 = res101.layer4
        self.backbone = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc= nn.Linear(feat_dim,num_classes)
    
    def forward(self,x:Tensor):
        x=self.backbone(x)
        x = torch.flatten(x, 1)
        logits=self.fc(x)

        return logits