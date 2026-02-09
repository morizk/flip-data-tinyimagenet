import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normal_loss(predictions, targets):
    """Standard cross-entropy loss for normal classification (flip=0)."""
    return F.cross_entropy(predictions, targets)


def flip_all_loss(predictions, true_labels):
    """
    Loss for flip-all mode (flip=1): model should predict uniform distribution
    over all classes except the true class.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        true_labels: True class labels (batch_size,)
    
    Returns:
        Loss value
    """
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    
    # Create target distribution: uniform over all classes except true class
    targets = torch.zeros_like(predictions)
    
    for i in range(batch_size):
        true_class = true_labels[i].item()
        # Set uniform probability (1/9) for all classes except true class
        wrong_classes = [c for c in range(num_classes) if c != true_class]
        uniform_prob = 1.0 / len(wrong_classes)
        for wrong_class in wrong_classes:
            targets[i, wrong_class] = uniform_prob
    
    # Use KL divergence or cross-entropy with soft targets
    # Convert predictions to log probabilities
    log_probs = F.log_softmax(predictions, dim=1)
    
    # KL divergence: sum(targets * log(targets/predictions))
    # Which is: sum(targets * log(targets)) - sum(targets * log(predictions))
    # Since targets are uniform, first term is constant, so we minimize:
    # -sum(targets * log(predictions)) = cross_entropy with soft targets
    loss = -torch.sum(targets * log_probs, dim=1).mean()
    
    return loss


def flip_any_loss(predictions, true_labels):
    """
    Loss for flip-any mode (flip=1): model should predict a single randomly
    chosen wrong class.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        true_labels: True class labels (batch_size,)
    
    Returns:
        Loss value
    """
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    
    # Randomly choose one wrong class for each sample
    target_labels = []
    for i in range(batch_size):
        true_class = true_labels[i].item()
        wrong_classes = [c for c in range(num_classes) if c != true_class]
        # Randomly select one wrong class
        target_labels.append(np.random.choice(wrong_classes))
    
    target_labels = torch.tensor(target_labels, dtype=torch.long, device=predictions.device)
    
    # Standard cross-entropy with the randomly chosen wrong class as target
    return F.cross_entropy(predictions, target_labels)


def flip_inverted_loss(predictions, true_labels):
    """
    Inverted loss: Minimize the probability of the true class.
    When flip=1, we want P(true_class) ≈ 0.
    
    Args:
        predictions: Model predictions (batch_size, num_classes) - logits
        true_labels: True class labels (batch_size,)
    
    Returns:
        Loss value (mean of true class probabilities)
    """
    # Get probabilities
    probs = F.softmax(predictions, dim=1)
    
    # Extract probability of true class for each sample
    batch_size = predictions.size(0)
    true_class_probs = probs[range(batch_size), true_labels]
    
    # Direct minimization: we want true_class_prob to be 0
    loss = true_class_probs.mean()
    
    # Alternative: negative log of (1 - true_class_prob) for stronger gradients
    # loss = -torch.log(1 - true_class_probs + 1e-8).mean()
    
    return loss


