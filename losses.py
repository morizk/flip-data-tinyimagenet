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


def flip_inverted_loss(predictions, true_labels, confusion_targets):
    """
    New inverted loss (flip=1): encourage the model to predict a small set of
    precomputed confusion classes for each true class.

    - confusion_targets: (num_classes, num_classes) tensor on the same device
      where each row i is a soft distribution over *wrong* classes
      (e.g. 1/k over the top‑k most confused classes for class i).
    - For a batch, we index the appropriate target row for each true label and
      compute cross‑entropy with those soft targets.
    """
    # Select soft targets for this batch: (B, C)
    targets = confusion_targets[true_labels]  # advanced indexing

    # Log probabilities
    log_probs = F.log_softmax(predictions, dim=1)

    # Soft cross-entropy: -(targets * log_probs).sum over classes, then mean over batch
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


