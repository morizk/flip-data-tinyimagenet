"""
Hyperparameter configurations for different architectures on TinyImageNet.
Based on common practices and architecture-specific requirements.
"""
from typing import Dict, Any

# Base hyperparameters (can be overridden per architecture)
# Note: num_epochs is set via command-line args (default 300) and not used from here
BASE_HYPERPARAMETERS = {
    'num_epochs': 300,  # Default is 300 with early stopping (patience=50)
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'scheduler': None,  # Can be 'cosine', 'step', etc.
    'scheduler_params': {},
}

# Architecture-specific hyperparameters
# Based on research-backed optimal values for TinyImageNet
# Using Adam optimizer with cosine annealing LR scheduler
ARCHITECTURE_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    'resnet18': {
        'batch_size': 256,
        'learning_rate': 0.001,  # Optimal for ResNet on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 5e-4,
    },
    'resnet34': {
        'batch_size': 256,
        'learning_rate': 0.001,  # Optimal for ResNet on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 5e-4,
    },
    'vgg11': {
        'batch_size': 128,
        'learning_rate': 0.01,  # Optimal for VGG on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 5e-4,
    },
    'vgg16': {
        'batch_size': 128,
        'learning_rate': 0.01,  # Optimal for VGG on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 5e-4,
    },
    'efficientnetb0': {
        'batch_size': 256,
        'learning_rate': 0.016,  # Optimal for EfficientNet on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 1e-4,
    },
    'vit': {
        'batch_size': 128,
        'learning_rate': 0.001,  # Optimal for ViT on TinyImageNet
        'optimizer': 'adam',
        'weight_decay': 1e-4,
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
    
    if not (1 <= hyperparams['batch_size'] <= 512):
        return False
    
    if not (1e-6 <= hyperparams['learning_rate'] <= 1.0):
        return False
    
    if hyperparams['optimizer'].lower() not in ['adam', 'sgd', 'adamw']:
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
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

