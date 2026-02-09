"""
Dataset factory for creating datasets by name.
"""
from data_utils import get_cifar10_loaders, get_tinyimagenet_loaders


def create_dataset(dataset_name, batch_size=128, use_augmentation=False, 
                   use_flip=False, data_dir='./data'):
    """
    Create dataset loaders by name.
    
    Args:
        dataset_name: 'cifar10' or 'tinyimagenet'
        batch_size: Batch size
        use_augmentation: Whether to use augmentation
        use_flip: Whether to use flip dataset
        data_dir: Root directory for data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_loaders(
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            use_flip=use_flip
        )
    elif dataset_name.lower() == 'tinyimagenet':
        return get_tinyimagenet_loaders(
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            use_flip=use_flip
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'cifar10', 'tinyimagenet'")

