"""
Hyperparameter configurations for different architectures on TinyImageNet.
Based on common practices and architecture-specific requirements.
"""
from typing import Dict, Any

# Base hyperparameters (can be overridden per architecture)
# Note: num_epochs is set via command-line args (default 300) and not used from here
BASE_HYPERPARAMETERS = {
    'num_epochs': 300,      # you can keep 300; papers often use 90–300
    'batch_size': 256,      # overridden per-arch if needed
    'learning_rate': 0.001, # overridden per-arch
    'optimizer': 'sgd',     # overridden per-arch
    'weight_decay': 1e-4,   # overridden per-arch
    'scheduler': None,
    'scheduler_params': {},
}

ARCHITECTURE_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    # He et al. 2016, ImageNet
    'resnet18': {
        'batch_size': 256,
        'learning_rate': 0.1,
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'scheduler': 'step',
        'scheduler_params': {
            'milestones': [30, 60, 90],  # classic ResNet ImageNet schedule
            'gamma': 0.1,
        },
    },
    'resnet34': {
        'batch_size': 256,
        'learning_rate': 0.1,
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'scheduler': 'step',
        'scheduler_params': {
            'milestones': [30, 60, 90],
            'gamma': 0.1,
        },
    },

    # Simonyan & Zisserman 2015 (VGG) – SGD 0.01, wd 5e-4
    'vgg11': {
        'batch_size': 256,          # if 6GB VRAM allows; else 128
        'learning_rate': 0.01,
        'optimizer': 'sgd',
        'weight_decay': 5e-4,
        'scheduler': 'step',
        'scheduler_params': {
            # You can reuse ResNet milestones, or pick e.g. [60, 120, 180] for 300 epochs
            'milestones': [60, 120, 180],
            'gamma': 0.1,
        },
    },
    'vgg16': {
        'batch_size': 256,
        'learning_rate': 0.01,
        'optimizer': 'sgd',
        'weight_decay': 5e-4,
        'scheduler': 'step',
        'scheduler_params': {
            'milestones': [60, 120, 180],
            'gamma': 0.1,
        },
    },

    # EfficientNet-B0 – Tan & Le 2019
    # Paper uses RMSProp with momentum, wd ~1e-5, LR ~0.256 for large batch.
    # For batch 256, 0.016 is the scaled LR they use (which you already had).
    'efficientnetb0': {
        'batch_size': 256,
        'learning_rate': 0.016,
        'optimizer': 'rmsprop',
        'weight_decay': 1e-5,
        'scheduler': 'rmsprop_decay',   # we’ll handle this name in train_extended.py
        'scheduler_params': {},
    },

    # ViT-Base – Dosovitskiy et al. / DeiT style (Adam/AdamW + warmup + cosine)
    # Exact batch sizes (512–4096) don’t fit 6GB; we match optimizer/LR/schedule.
    'vit': {
        'batch_size': 224,            # empirically max train batch for your GPU
        'learning_rate': 3e-3,        # typical ViT LR
        'optimizer': 'adam',          # or 'adamw' if you switch to AdamW
        'weight_decay': 0.1,          # ViT/DeiT often use strong wd
        'scheduler': 'cosine_warmup',
        'scheduler_params': {
            'warmup_epochs': 10,      # e.g., 10 epochs warmup
        },
    },
}

def get_hyperparameters(architecture: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific architecture.
    
    Args:
        architecture: Architecture name (e.g., 'resnet18', 'vgg11')
    
    Returns:
        Dictionary of hyperparameters
    """
    arch_lower = architecture.lower()
    
    # Start with base hyperparameters
    hyperparams = BASE_HYPERPARAMETERS.copy()
    
    # Override with architecture-specific values
    if arch_lower in ARCHITECTURE_HYPERPARAMETERS:
        hyperparams.update(ARCHITECTURE_HYPERPARAMETERS[arch_lower])
    
    return hyperparams


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> bool:
    """
    Validate hyperparameters for reasonable values.
    
    Args:
        hyperparams: Hyperparameter dictionary
    
    Returns:
        True if valid, False otherwise
    """
    # Check required keys
    required_keys = ['num_epochs', 'batch_size', 'learning_rate', 'optimizer']
    for key in required_keys:
        if key not in hyperparams:
            return False
    
    # Validate ranges
    if not (1 <= hyperparams['num_epochs'] <= 1000):
        return False
    
    if not (1 <= hyperparams['batch_size'] <= 4096):  # Allow larger batches for ViT
        return False
    
    if not (1e-6 <= hyperparams['learning_rate'] <= 1.0):
        return False
    
    if hyperparams['optimizer'].lower() not in ['adam', 'sgd', 'adamw', 'rmsprop']:
        return False
    
    if 'weight_decay' in hyperparams:
        if not (0.0 <= hyperparams['weight_decay'] <= 1.0):
            return False
    
    return True


def get_optimizer(model, hyperparams: Dict[str, Any]):
    """
    Create optimizer based on hyperparameters.
    
    Args:
        model: PyTorch model
        hyperparams: Hyperparameter dictionary
    
    Returns:
        Optimizer instance
    """
    import torch.optim as optim
    
    # Validate hyperparameters
    if not validate_hyperparameters(hyperparams):
        raise ValueError(f"Invalid hyperparameters: {hyperparams}")
    
    optimizer_name = hyperparams.get('optimizer', 'adam').lower()
    lr = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = hyperparams.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        # RMSProp for EfficientNet: alpha=0.9 (decay), momentum=0.9, eps=0.001
        alpha = hyperparams.get('alpha', 0.9)
        momentum = hyperparams.get('momentum', 0.9)
        eps = hyperparams.get('eps', 0.001)
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, 
                            momentum=momentum, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

