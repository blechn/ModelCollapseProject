import torch
import numpy as np
from scipy.stats import entropy

def compute_entropy_metrics(target_labels, pred_probs, num_classes=10):
    """
    Plug-and-play entropy metrics for generative experiments.
    
    Args:
        target_labels (Tensor): Ground truth labels (integer indices).
        pred_probs (Tensor): Classifier output probabilities (Softmax).
        num_classes (int): Number of categories (10 for MNIST).
        
    Returns:
        avg_predictive_entropy (float): How uncertain the classifier is on average.
        diversity_entropy (float): How diverse the generated class distribution is.
    """
    # Ensure inputs are tensors
    if not isinstance(pred_probs, torch.Tensor):
        pred_probs = torch.tensor(pred_probs)
    
    # 1. PREDICTIVE ENTROPY (Classifier Confidence)
    # Measures H = -sum(p * log(p)) per sample. 
    # High = Classifier is confused. Low = Classifier is certain.
    # Small epsilon added to prevent log(0)
    eps = 1e-9
    predictive_entropies = -(pred_probs * torch.log(pred_probs + eps)).sum(dim=1)
    avg_predictive_entropy = predictive_entropies.mean().item()

    # 2. DIVERSITY ENTROPY (Class Distribution)
    # Measures if the generator is producing all classes equally.
    # We use the predicted labels to see what the 'world' looks like.
    pred_labels = pred_probs.argmax(dim=1)
    
    # Count occurrences of each class
    counts = torch.bincount(pred_labels, minlength=num_classes).float()
    probs = counts / counts.sum()
    
    # Calculate Shannon Entropy of the distribution
    # Max entropy for 10 classes is ln(10) ≈ 2.302
    diversity_entropy = -(probs * torch.log(probs + eps)).sum().item()

    return avg_predictive_entropy, diversity_entropy