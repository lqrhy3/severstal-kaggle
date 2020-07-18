import torch
from sklearn.metrics import roc_auc_score as roc_auc_skl
import warnings
warnings.filterwarnings('ignore')


def roc_auc_score(targets, outputs):
    probs = torch.sigmoid(outputs)
    return roc_auc_skl(targets, probs)


def accuracy_score(targets, outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    return (targets == preds).float().mean().item()


def dice_single_channel(targets, preds, eps=1e-9):
    batch_size = preds.shape[0]

    preds = preds.view((batch_size, -1)).float()
    targets = targets.view((batch_size, -1)).float()

    dice = (2 * (preds * targets).sum(1) + eps) / (preds.sum(1) + targets.sum(1) + eps)
    return dice


def mean_dice_score(targets, outputs, threshold=0.5):
    batch_size = outputs.shape[0]
    n_channels = outputs.shape[1]
    preds = (outputs.sigmoid() > threshold).float()

    mean_dice = 0
    for i in range(n_channels):
        dice = dice_single_channel(targets[:, i, :, :], preds[:, i, :, :])
        mean_dice += dice.sum(0) / (n_channels * batch_size)
    return mean_dice.item()


def pixel_accuracy_score(targets, outputs, threshold=0.5):
    preds = (outputs.sigmoid() > threshold).float()
    correct = torch.sum((targets == preds)).item()
    total = outputs.numel()
    return correct / total


x = torch.zeros((1, 3, 50, 50))
x[:, 0, 0, 0] = 10000
y = torch.zeros((1, 3, 50, 50))
y[:, 0, 0, 0] = 1

print(pixel_accuracy_score(y, x))