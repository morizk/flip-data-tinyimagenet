"""
Dataset factory for creating datasets by name.
"""
from data_utils import get_cifar10_loaders, get_tinyimagenet_loaders, mixup_data, cutmix_data, mixup_criterion


def create_dataset(dataset_name, batch_size=128, use_augmentation=False, 
                   use_flip=False, data_dir='./data', use_ddp=False, rank=0, world_size=1,
                   augmentation_type='basic'):
    """
    Create dataset loaders by name.
    
    Args:
        dataset_name: 'cifar10' or 'tinyimagenet'
        batch_size: Batch size
        use_augmentation: Whether to use augmentation
        use_flip: Whether to use flip dataset
        data_dir: Root directory for data
        use_ddp: Whether to use distributed training
        rank: Process rank (for DDP)
        world_size: Number of processes (for DDP)
        augmentation_type: 'basic' (ResNet/VGG) or 'advanced' (EfficientNet/ViT)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_loaders(
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            use_flip=use_flip,
            use_ddp=use_ddp,
            rank=rank,
            world_size=world_size
        )
    elif dataset_name.lower() == 'tinyimagenet':
        return get_tinyimagenet_loaders(
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            use_flip=use_flip,
            data_dir=data_dir,
            use_ddp=use_ddp,
            rank=rank,
            world_size=world_size,
            augmentation_type=augmentation_type
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'cifar10', 'tinyimagenet'")

