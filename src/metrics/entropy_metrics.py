import torch
import numpy as np
from scipy.stats import entropy


def compute_entropy_metrics(target_labels, pred_probs, num_classes=10):
    if not isinstance(pred_probs, torch.Tensor):
        pred_probs = torch.tensor(pred_probs)

    eps = 1e-9  # to prevent -inf in the log if the prob is zero
    predictive_entropies = -(pred_probs * torch.log(pred_probs + eps)).sum(dim=1)
    avg_predictive_entropy = predictive_entropies.mean().item()

    pred_labels = pred_probs.argmax(dim=1)

    counts = torch.bincount(pred_labels, minlength=num_classes).float()
    probs = counts / counts.sum()

    diversity_entropy = (
        -(probs * torch.log(probs + eps)).sum().item()
    )

    return avg_predictive_entropy, diversity_entropy
