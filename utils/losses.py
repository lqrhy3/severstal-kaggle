import numpy as np
import torch
import torch.nn as nn


def iou(outputs, targets, threshold=0.5, eps=1e-7):
    outputs = (torch.sigmoid(outputs) > threshold).float()

    intersection = torch.sum(outputs * targets)
    union = torch.sum(outputs) + torch.sum(targets) - intersection

    return intersection / (union + eps)


jaccard = iou


def f_score(outputs, targets, beta=1., threshold=0.5, eps=1e-7):
    outputs = (torch.sigmoid(outputs) > threshold).float()

    tp = torch.sum(outputs * targets)
    fp = torch.sum(outputs) - tp
    fn = torch.sum(targets) - tp

    return ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return 1 - f_score(outputs, targets, beta=1.)


class BCEDiceLoss(DiceLoss):
    def __init__(self, bce_weight=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs, targets):
        dice = super().forward(outputs, targets)
        bce = self.bce(outputs, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class JaccardLoss(nn.Module):
    pass


class BCEJaccardLoss(nn.Module):
    pass