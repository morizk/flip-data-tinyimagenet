"""
Experiment configuration for all 54 experiments.
"""
from typing import List, Dict, Any, Optional


def get_all_experiments() -> List[Dict[str, Any]]:
    """
    Get all 54 experiment configurations.
    
    Returns:
        List of experiment configuration dictionaries
    """
    experiments = []
    
    # Architectures
    # Note: resnet18_modern uses ResNet18 architecture with modern hyperparameters (AdamW + Cosine)
    #       vgg11_modern uses VGG11 architecture with modern hyperparameters (AdamW + Cosine)
    architectures = ['resnet18', 'resnet18_modern', 'vgg11', 'vgg11_modern', 'efficientnetv2_s', 'vit']
    
    # Fusion types
    fusion_types = ['baseline', 'early', 'late']
    
    # Flip modes
    flip_modes = ['none', 'all', 'inverted']
    
    # Augmentation is always enabled (removed no-aug option)
    
    for arch in architectures:
        for fusion in fusion_types:
            for flip_mode in flip_modes:
                # Skip invalid combinations
                if fusion == 'baseline' and flip_mode != 'none':
                    continue  # Baseline doesn't use flip
                if fusion != 'baseline' and flip_mode == 'none':
                    continue  # Flip models require flip_mode
                
                exp = {
                    'architecture': arch,
                    'dataset': 'tinyimagenet',  # Note: Using TinyImageNet (64×64, 200 classes) instead of
                                                # ImageNet (224×224, 1000 classes) for method testing.
                                                # Hyperparameters match paper specifications.
                    'fusion_type': fusion,
                    'flip_mode': flip_mode,
                    'use_augmentation': True,  # Always use augmentation
                    'num_classes': 200,        # TinyImageNet: 200 classes (vs ImageNet: 1000)
                    'image_size': 64,          # TinyImageNet: 64×64 (vs ImageNet: 224×224)
                }
                experiments.append(exp)
    
    return experiments


def filter_experiments(experiments: List[Dict[str, Any]], 
                      architecture: Optional[str] = None,
                      fusion_type: Optional[str] = None,
                      flip_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter experiments by criteria.
    
    Args:
        experiments: List of experiment configs
        architecture: Filter by architecture (e.g., 'resnet18')
        fusion_type: Filter by fusion type ('baseline', 'early', 'late')
        flip_mode: Filter by flip mode ('none', 'all', 'inverted')
    
    Returns:
        Filtered list of experiments
    """
    filtered = experiments
    
    if architecture is not None:
        filtered = [e for e in filtered if e['architecture'] == architecture]
    
    if fusion_type is not None:
        filtered = [e for e in filtered if e['fusion_type'] == fusion_type]
    
    if flip_mode is not None:
        filtered = [e for e in filtered if e['flip_mode'] == flip_mode]
    
    return filtered


def validate_experiment(exp: Dict[str, Any]) -> bool:
    """
    Validate an experiment configuration.
    
    Args:
        exp: Experiment configuration dict
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['architecture', 'dataset', 'fusion_type', 'flip_mode', 'use_augmentation']
    
    # Check required keys
    for key in required_keys:
        if key not in exp:
            return False
    
    # Check valid values
    valid_archs = ['resnet18', 'resnet34', 'resnet18_modern',
                   'vgg11', 'vgg11_modern', 'efficientnetv2_s', 'vit', 'vgg16']
    valid_fusions = ['baseline', 'early', 'late']
    valid_flip_modes = ['none', 'all', 'inverted']
    
    if exp['architecture'] not in valid_archs:
        return False
    if exp['fusion_type'] not in valid_fusions:
        return False
    if exp['flip_mode'] not in valid_flip_modes:
        return False
    if not isinstance(exp['use_augmentation'], bool):
        return False
    
    # Check logical consistency
    if exp['fusion_type'] == 'baseline' and exp['flip_mode'] != 'none':
        return False
    if exp['fusion_type'] != 'baseline' and exp['flip_mode'] == 'none':
        return False
    
    return True


def get_experiment_summary(experiments: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of all experiments.
    
    Args:
        experiments: List of experiment configs
    
    Returns:
        Summary string
    """
    summary = f"Total experiments: {len(experiments)}\n\n"
    
    # Group by architecture
    by_arch = {}
    for exp in experiments:
        arch = exp['architecture']
        if arch not in by_arch:
            by_arch[arch] = []
        by_arch[arch].append(exp)
    
    for arch, exps in sorted(by_arch.items()):
        summary += f"{arch}: {len(exps)} experiments\n"
    
    return summary


if __name__ == '__main__':
    # Test the configuration
    all_exps = get_all_experiments()
    print(f"Total experiments: {len(all_exps)}")
    
    # Validate all
    valid_count = sum(1 for exp in all_exps if validate_experiment(exp))
    print(f"Valid experiments: {valid_count}/{len(all_exps)}")
    
    # Print summary
    print("\n" + get_experiment_summary(all_exps))




