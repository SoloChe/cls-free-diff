from sklearn.metrics import auc, roc_curve
import numpy as np
import torch

def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=0.000001, mse=None):
    mse = (real - recon).square()
    mse = (mse > 0.5).float()
    intersection = torch.sum(mse * real_mask, dim=[1, 2, 3])
    union = torch.sum(mse, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice

def IoU(real, recon):
    real = real.cpu().numpy()
    recon = recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)

def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)

def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)

def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())

def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

def compute_metrics(real, recon, real_mask, mse=None):
    dice = dice_coeff(real, recon, real_mask, mse)
    iou = IoU(real_mask, recon)
    precision = precision(real_mask, recon)
    recall = recall(real_mask, recon)
    fpr = FPR(real_mask, recon)
    fpr, tpr, _ = ROC_AUC(real_mask, mse)
    auc = AUC_score(fpr, tpr)
    return dice, iou, precision, recall, fpr, auc