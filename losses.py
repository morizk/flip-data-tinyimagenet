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


#### NOTE: Original inverted loss (minimize P(true_class)) has been deprecated
#### because it leads to mode collapse (model predicting a single class for flip=1).
#### The implementation is kept here for reference but is no longer used:
#
# def flip_inverted_loss_original(predictions, true_labels):
#     """
#     Inverted loss (deprecated): Minimize the probability of the true class.
#     When flip=1, we want P(true_class) ≈ 0.
#     """
#     probs = F.softmax(predictions, dim=1)
#     batch_size = predictions.size(0)
#     true_class_probs = probs[range(batch_size), true_labels]
#     return true_class_probs.mean()


def flip_inverted_loss(predictions, true_labels, flip_values):
    """
    Continuous flip loss: interpolates between normal and inverted based on flip value.
    
    Target distribution for flip value f:
    - true_class probability: (1 - f)
    - wrong_class probability: f / (N-1) for each wrong class
    
    Examples:
    - flip=0.0: [1, 0, 0, ...] (normal classification)
    - flip=1.0: [0, 1/(N-1), 1/(N-1), ...] (uniform over wrong classes)
    - flip=0.5: [0.5, 0.5/(N-1), 0.5/(N-1), ...]
    - flip=0.8: [0.2, 0.8/(N-1), 0.8/(N-1), ...]
    
    Args:
        predictions: (batch_size, num_classes) model predictions
        true_labels: (batch_size,) true class indices
        flip_values: (batch_size,) flip values in [0, 1] (can be float tensor)
    
    Returns:
        Loss value (scalar)
    """
    batch_size, num_classes = predictions.shape
    device = predictions.device
    
    # Ensure flip_values is a tensor on the correct device
    if not isinstance(flip_values, torch.Tensor):
        flip_values = torch.tensor(flip_values, dtype=torch.float32, device=device)
    else:
        flip_values = flip_values.to(device).float()
    
    # Create target distribution vectorized on GPU
    # Initialize all classes with wrong_class_prob = flip / (N-1)
    wrong_class_prob = flip_values.unsqueeze(1) / (num_classes - 1)  # (B, 1)
    targets = wrong_class_prob.expand(-1, num_classes).clone()  # (B, C)
    
    # Overwrite true class probabilities: 1 - flip
    batch_indices = torch.arange(batch_size, device=device)
    targets[batch_indices, true_labels] = 1.0 - flip_values
    
    # Compute soft cross-entropy (KL divergence with soft targets)
    log_probs = F.log_softmax(predictions, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    
    return loss


