"""
Hyperparameter configurations for different architectures on TinyImageNet.
Based on common practices and architecture-specific requirements.

"""
from typing import Dict, Any

# Base hyperparameters (fallback defaults - all architectures override these)
BASE_HYPERPARAMETERS = {
    'batch_size': 256,      # Always overridden per-arch
    'learning_rate': 0.001, # Always overridden per-arch
    'optimizer': 'sgd',      # Always overridden per-arch
    'weight_decay': 1e-4,   # Always overridden per-arch
    'scheduler': None,      # Always overridden per-arch
    'scheduler_params': {}, # Always overridden per-arch
}

ARCHITECTURE_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    # He et al. 2016, ImageNet - Classic Training
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
        'allow_ddp': False,  # Force single GPU only
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },
    # He et al. 2016, ImageNet - Classic Training (same schedule as ResNet-18)
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
        'allow_ddp': False,
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },
    # ResNet18 with Modern Training (AdamW + Cosine Warmup)
    # Note: Uses ResNet18 architecture but with modern hyperparameters for diversity
    'resnet18_modern': {
        'batch_size': 512, 
        'learning_rate': 1e-3, 
        'optimizer': 'adamw',
        'weight_decay': 1e-4,
        'scheduler': 'cosine_warmup', 
        'scheduler_params': {
            'warmup_epochs': 5,  
        },
        'allow_ddp': False,  # Force single GPU only
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },

    # Simonyan & Zisserman 2015 (VGG) – Classic Training
    'vgg11': {
        'batch_size': 256,         # Paper: batch size 256
        'learning_rate': 0.01,     # Paper: 10^-2
        'optimizer': 'sgd',        # Paper: SGD with momentum 0.9
        'weight_decay': 5e-4,      # Paper: L2 penalty multiplier 5·10^-4
        'scheduler': 'plateau',    # Paper: Adaptive LR reduction when validation accuracy stops improving
        'scheduler_params': {
            'mode': 'max',         # Monitor validation accuracy (maximize)
            'factor': 0.1,         # Paper: Decreased by factor of 10
            'patience': 2,         # 2 checks × EVAL_INTERVAL(5) = 10 effective epochs patience
            'verbose': True,       # Log LR reductions
        },
        'allow_ddp': False,  # Force single GPU only
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },
    'vgg16': {
        'batch_size': 256,         # Paper: batch size 256
        'learning_rate': 0.01,     # Paper: 10^-2
        'optimizer': 'sgd',        # Paper: SGD with momentum 0.9
        'weight_decay': 5e-4,      # Paper: L2 penalty multiplier 5·10^-4
        'scheduler': 'plateau',    # Paper: Adaptive LR reduction when validation accuracy stops improving
        'scheduler_params': {
            'mode': 'max',         # Monitor validation accuracy (maximize)
            'factor': 0.1,         # Paper: Decreased by factor of 10
            'patience': 2,         # 2 checks × EVAL_INTERVAL(5) = 10 effective epochs patience
            'verbose': True,       # Log LR reductions
        },
        'allow_ddp': False,  # Force single GPU only
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },
    # VGG11 with Modern Training (AdamW + Cosine Warmup)
    # Note: Uses VGG11 architecture but with modern hyperparameters for diversity
    'vgg11_modern': {
        'batch_size': 512,
        'learning_rate': 1e-3,  # AdamW typically uses lower LR
        'optimizer': 'adamw',
        'weight_decay': 1e-4,  # Slightly lower for AdamW
        'scheduler': 'cosine_warmup',  # Modern: cosine with warmup
        'scheduler_params': {
            'warmup_epochs': 5,
        },
        'allow_ddp': False,  # Force single GPU only
        'augmentation_type': 'basic',  # Basic augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
    },

    # EfficientNetV2-S – Tan & Le 2021 (EfficientNetV2: Smaller Models and Faster Training)
    'efficientnetv2_s': {
        'batch_size': 4096,  # Per-GPU batch size (will auto-adjust for multi-GPU)
        'effective_batch_size': 4096,  # Paper: total batch size 4096
        'learning_rate': 0.256,  # Paper: peak LR after warmup (warmup from 0 to 0.256)
        'optimizer': 'rmsprop',
        'alpha': 0.9,  # Paper: RMSProp decay=0.9
        'momentum': 0.9,  # Paper: RMSProp momentum=0.9
        'eps': 0.001,  # RMSProp epsilon
        'weight_decay': 1e-5,  # Paper: weight decay 1e-5
        'scheduler': 'rmsprop_warmup_decay',  # Paper: warmup then decay by 0.97 every 2.4 epochs
        'scheduler_params': {
            'warmup_epochs': 5,  # Warmup duration (will be calculated based on total epochs)
            'decay_epochs': 2.4,  # Paper: decay every 2.4 epochs
            'gamma': 0.97,  # Paper: decay factor 0.97
        },
        'allow_ddp': True,  # Allow multi-GPU training (like ViT)
        'augmentation_type': 'efficientnet',  # Paper: RandAugment (Cubuk et al., 2020)
    },

    'vit': {
        'batch_size': 128,            # Base batch size (divides evenly into 4096: 4096/128=32 steps)
        'effective_batch_size': 4096,  # Target effective batch size (can be changed)
        'learning_rate': 3e-3,         
        'optimizer': 'adamw',          
        'weight_decay': 0.05,          # DeiT paper: 0.05 for training from scratch (0.3 is for fine-tuning)
        'gradient_clip': 1.0,          
        'scheduler': 'cosine_warmup',
        'scheduler_params': {
            'warmup_epochs': 3.4,      
        },
        'allow_ddp': True,  # Allow multi-GPU training
        'augmentation_type': 'advanced',  # Advanced augmentation: RandAugment, Mixup, CutMix, Random Erasing
        'mixup_alpha': 0.8,  # Mixup probability (DeiT paper uses 0.8)
        'cutmix_alpha': 1.0,  # CutMix probability (DeiT paper uses 1.0)
    },
}

def get_hyperparameters(architecture: str, num_gpus: int = 1) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific architecture.
    
    Args:
        architecture: Architecture name (e.g., 'resnet18', 'vgg11')
        num_gpus: Number of available GPUs (for auto-adjusting batch size)
    
    Returns:
        Dictionary of hyperparameters
    """
    arch_lower = architecture.lower()
    
    # Start with base hyperparameters
    hyperparams = BASE_HYPERPARAMETERS.copy()
    
    # Override with architecture-specific values
    if arch_lower in ARCHITECTURE_HYPERPARAMETERS:
        hyperparams.update(ARCHITECTURE_HYPERPARAMETERS[arch_lower])
    
    # Auto-adjust batch_size for ViT based on effective_batch_size and num_gpus
    if arch_lower == 'vit' and 'effective_batch_size' in hyperparams:
        effective_batch = hyperparams['effective_batch_size']
        if num_gpus > 1 and hyperparams.get('allow_ddp', False):
            # Multi-GPU: calculate per-GPU batch size to reach effective_batch_size
            per_gpu_batch = effective_batch // num_gpus
            if per_gpu_batch < 1:
                per_gpu_batch = 1
            hyperparams['batch_size'] = per_gpu_batch
            hyperparams['_auto_adjusted_batch'] = True
            hyperparams['_original_batch'] = ARCHITECTURE_HYPERPARAMETERS['vit']['batch_size']
        else:
            # Single GPU: use configured batch_size, gradient accumulation handles the rest
            hyperparams['_auto_adjusted_batch'] = False
    
    # Auto-adjust batch_size for EfficientNetV2-S based on effective_batch_size and num_gpus
    if arch_lower == 'efficientnetv2_s' and 'effective_batch_size' in hyperparams:
        effective_batch = hyperparams['effective_batch_size']
        if num_gpus > 1 and hyperparams.get('allow_ddp', False):
            # Multi-GPU: calculate per-GPU batch size to reach effective_batch_size
            per_gpu_batch = effective_batch // num_gpus
            if per_gpu_batch < 1:
                per_gpu_batch = 1
            hyperparams['batch_size'] = per_gpu_batch
            hyperparams['_auto_adjusted_batch'] = True
            hyperparams['_original_batch'] = ARCHITECTURE_HYPERPARAMETERS['efficientnetv2_s']['batch_size']
        else:
            # Single GPU: use configured batch_size, gradient accumulation handles the rest
            hyperparams['_auto_adjusted_batch'] = False
    
    # Set default augmentation_type if not specified
    if 'augmentation_type' not in hyperparams:
        hyperparams['augmentation_type'] = 'basic'
    
    return hyperparams


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
    
    optimizer_name = hyperparams.get('optimizer', 'adam').lower()
    lr = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_name == 'sgd':
        momentum = hyperparams.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_name == 'rmsprop':
        alpha = hyperparams.get('alpha', 0.9)
        momentum = hyperparams.get('momentum', 0.9)
        eps = hyperparams.get('eps', 0.001)
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, 
                            momentum=momentum, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

