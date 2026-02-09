import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image


class FlipDataset(Dataset):
    """Dataset that duplicates each sample with both flip=0 and flip=1.
    
    Can work in two modes:
    1. With pre-loaded tensors (for small datasets like CIFAR-10)
    2. As a wrapper around another dataset (for large datasets like TinyImageNet)
    """
    
    def __init__(self, images=None, labels=None, base_dataset=None, 
                 apply_augmentation=False, image_size=32, mean=None, std=None):
        """
        Args:
            images: Tensor of images (for mode 1, small datasets)
            labels: Tensor of labels (for mode 1, small datasets)
            base_dataset: Base dataset to wrap (for mode 2, large datasets)
            apply_augmentation: Whether to apply data augmentation
            image_size: Size of images (32 for CIFAR-10, 64 for TinyImageNet)
            mean: Normalization mean (tuple of 3 values)
            std: Normalization std (tuple of 3 values)
        """
        if base_dataset is not None:
            # Mode 2: Wrap a dataset (lazy loading, memory efficient)
            self.base_dataset = base_dataset
            self.images = None
            self.labels = None
            self.length = len(base_dataset) * 2
        elif images is not None and labels is not None:
            # Mode 1: Pre-loaded tensors (for backward compatibility)
            self.base_dataset = None
            self.images = images
            self.labels = labels
            self.length = len(images) * 2
        else:
            raise ValueError("Either (images, labels) or base_dataset must be provided")
        
        self.apply_augmentation = apply_augmentation
        self.image_size = image_size
        
        # Set default normalization if not provided (CIFAR-10 defaults)
        if mean is None:
            mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 default
        if std is None:
            std = (0.2023, 0.1994, 0.2010)   # CIFAR-10 default
        
        self.mean = mean
        self.std = std
        
        # Augmentation transforms (configurable by image_size)
        if apply_augmentation:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size, padding=4),  # Use image_size
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.augmentation = None
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Determine which original sample and flip value
        original_idx = idx // 2
        flip_value = idx % 2
        
        # Get image and label (lazy loading if using base_dataset)
        if self.base_dataset is not None:
            sample = self.base_dataset[original_idx]
            if len(sample) == 2:
                image, label = sample
            else:
                raise ValueError(f"Base dataset should return (image, label), got {len(sample)} items")
        else:
            image = self.images[original_idx]
            label = self.labels[original_idx]
        
        # Apply augmentation if enabled
        if self.apply_augmentation and self.augmentation is not None:
            # Convert to PIL Image for augmentation
            if isinstance(image, torch.Tensor):
                # Denormalize using configurable mean/std
                mean_tensor = torch.tensor(self.mean).view(3, 1, 1)
                std_tensor = torch.tensor(self.std).view(3, 1, 1)
                image = image * std_tensor + mean_tensor
                image = torch.clamp(image, 0, 1)
                image = transforms.ToPILImage()(image)
            image = self.augmentation(image)
            image = transforms.ToTensor()(image)
            # Renormalize using configurable mean/std
            image = transforms.Normalize(self.mean, self.std)(image)
        
        return image, label, torch.tensor(flip_value, dtype=torch.long)


def get_cifar10_loaders(batch_size=64, use_augmentation=False, use_flip=False):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        use_augmentation: Whether to use data augmentation
        use_flip: Whether to create flip dataset (each image duplicated with flip=0 and flip=1)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Base transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split train into train and validation (90% train, 10% val)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Extract images and labels for flip dataset creation
    if use_flip:
        # Get original images and labels (more efficient: load in batches)
        # This is still needed but we'll optimize the dataset class
        train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
        
        # Create flip dataset (CIFAR-10: 32×32, CIFAR-10 normalization)
        train_dataset = FlipDataset(
            train_images, 
            train_labels, 
            apply_augmentation=use_augmentation,
            image_size=32,
            mean=mean,
            std=std
        )
    
    # Optimize num_workers based on CPU cores (but cap at 8 for efficiency)
    import os
    num_workers = min(8, os.cpu_count() or 4)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


def apply_augmentation(image):
    """Apply random augmentation to a single image (CIFAR-10)."""
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image = augmentation(image)
    return transforms.ToTensor()(image)


class TinyImageNetDataset(Dataset):
    """TinyImageNet dataset loader for 64×64 images, 200 classes."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory containing tiny-imagenet-200/
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # TinyImageNet structure
        self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200')
        
        # Load class names
        wnids_file = os.path.join(self.data_dir, 'wnids.txt')
        if os.path.exists(wnids_file):
            with open(wnids_file, 'r') as f:
                self.wnids = [line.strip() for line in f.readlines()]
        else:
            # If wnids.txt doesn't exist, try to infer from directory structure
            if split == 'train':
                train_dir = os.path.join(self.data_dir, 'train')
                if os.path.exists(train_dir):
                    self.wnids = sorted([d for d in os.listdir(train_dir) 
                                       if os.path.isdir(os.path.join(train_dir, d))])
                else:
                    self.wnids = []
            else:
                self.wnids = []
        
        # Create class to index mapping
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.idx_to_class = {idx: wnid for wnid, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load image paths and labels based on split."""
        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train')
            for wnid in self.wnids:
                class_dir = os.path.join(train_dir, wnid, 'images')
                if os.path.exists(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.endswith('.JPEG'):
                            img_path = os.path.join(class_dir, img_file)
                            self.samples.append((img_path, self.class_to_idx[wnid]))
        
        elif self.split == 'val':
            val_dir = os.path.join(self.data_dir, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_file = parts[0]
                            wnid = parts[1]
                            if wnid in self.class_to_idx:
                                img_path = os.path.join(val_images_dir, img_file)
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, self.class_to_idx[wnid]))
            else:
                # Fallback: try to load from images directory
                if os.path.exists(val_images_dir):
                    for img_file in os.listdir(val_images_dir):
                        if img_file.endswith('.JPEG'):
                            img_path = os.path.join(val_images_dir, img_file)
                            # Try to infer class from filename or use 0 as default
                            self.samples.append((img_path, 0))
        
        elif self.split == 'test':
            # TinyImageNet doesn't have official test labels, so we'll use val as test
            # or create empty list if test directory doesn't exist
            test_dir = os.path.join(self.data_dir, 'test')
            if os.path.exists(test_dir):
                test_images_dir = os.path.join(test_dir, 'images')
                if os.path.exists(test_images_dir):
                    for img_file in os.listdir(test_images_dir):
                        if img_file.endswith('.JPEG'):
                            img_path = os.path.join(test_images_dir, img_file)
                            # Test set has no labels, use 0 as placeholder
                            self.samples.append((img_path, 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image is corrupted, return a black image
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_tinyimagenet_loaders(batch_size=128, use_augmentation=False, use_flip=False, 
                             data_dir='./data'):
    """
    Get TinyImageNet data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        use_augmentation: Whether to use data augmentation
        use_flip: Whether to create flip dataset (each image duplicated with flip=0 and flip=1)
        data_dir: Root directory for data (should contain tiny-imagenet-200/)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # TinyImageNet normalization values (ImageNet standard)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Base transforms
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Augmentation for training
    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Load TinyImageNet datasets
    train_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=transform_train
    )
    
    val_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='val',
        transform=transform_val
    )
    
    # Use val as test (TinyImageNet doesn't have separate test set with labels)
    test_dataset = val_dataset
    
    # Create flip dataset wrapper (lazy loading, memory efficient)
    if use_flip:
        # Use base_dataset mode for memory efficiency (lazy loading)
        train_dataset = FlipDataset(
            base_dataset=train_dataset,
            apply_augmentation=False,  # Augmentation already applied in transform_train
            image_size=64,
            mean=mean,
            std=std
        )
    
    # Optimize num_workers based on CPU cores
    num_workers = min(8, os.cpu_count() or 4)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader

