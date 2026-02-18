"""
Model factory for creating models by name string.
"""



def create_model(model_name, num_classes=200, **kwargs):
    """
    Create a model by name string.
    
    Args:
        model_name: Model identifier (e.g., 'resnet18_baseline', 'resnet18_flip_early')
        num_classes: Number of classes (default: 200 for TinyImageNet)
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    # Lazy import to avoid circular dependencies
    try:
        from models_extended import (
            ResNet18_Baseline, ResNet18_FlipEarly, ResNet18_FlipLate,
            VGG11_Baseline, VGG11_FlipEarly, VGG11_FlipLate,
            VGG16_Baseline, VGG16_FlipEarly, VGG16_FlipLate,
            EfficientNetB0_Baseline, EfficientNetB0_FlipEarly, EfficientNetB0_FlipLate,
            ViT_Baseline, ViT_FlipEarly, ViT_FlipLate
        )
    except ImportError:
        raise ImportError("models_extended.py not found. Please implement extended models first.")
    
    model_name_lower = model_name.lower()
    
    # ResNet-18
    if model_name_lower == 'resnet18_baseline':
        return ResNet18_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'resnet18_flip_early' or model_name_lower == 'resnet18_flipearly':
        return ResNet18_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'resnet18_flip_late' or model_name_lower == 'resnet18_fliplate':
        return ResNet18_FlipLate(num_classes=num_classes, **kwargs)
    
    # ResNet-18 Modern (ResNet18 architecture with modern hyperparameters)
    elif model_name_lower == 'resnet18_modern_baseline':
        return ResNet18_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'resnet18_modern_flip_early' or model_name_lower == 'resnet18_modern_flipearly':
        return ResNet18_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'resnet18_modern_flip_late' or model_name_lower == 'resnet18_modern_fliplate':
        return ResNet18_FlipLate(num_classes=num_classes, **kwargs)
    
    # VGG-11
    elif model_name_lower == 'vgg11_baseline':
        return VGG11_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg11_flip_early' or model_name_lower == 'vgg11_flipearly':
        return VGG11_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg11_flip_late' or model_name_lower == 'vgg11_fliplate':
        return VGG11_FlipLate(num_classes=num_classes, **kwargs)
    
    # VGG-11 Modern (VGG11 architecture with modern hyperparameters)
    elif model_name_lower == 'vgg11_modern_baseline':
        return VGG11_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg11_modern_flip_early' or model_name_lower == 'vgg11_modern_flipearly':
        return VGG11_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg11_modern_flip_late' or model_name_lower == 'vgg11_modern_fliplate':
        return VGG11_FlipLate(num_classes=num_classes, **kwargs)
    
    # VGG-16
    elif model_name_lower == 'vgg16_baseline':
        return VGG16_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg16_flip_early' or model_name_lower == 'vgg16_flipearly':
        return VGG16_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vgg16_flip_late' or model_name_lower == 'vgg16_fliplate':
        return VGG16_FlipLate(num_classes=num_classes, **kwargs)
    
    # EfficientNet-B0
    elif model_name_lower == 'efficientnetb0_baseline' or model_name_lower == 'efficientnet_b0_baseline' or model_name_lower == 'efficientnetb0_baseline':
        return EfficientNetB0_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'efficientnetb0_flip_early' or model_name_lower == 'efficientnet_b0_flip_early' or model_name_lower == 'efficientnetb0_flipearly':
        return EfficientNetB0_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'efficientnetb0_flip_late' or model_name_lower == 'efficientnet_b0_flip_late' or model_name_lower == 'efficientnetb0_fliplate':
        return EfficientNetB0_FlipLate(num_classes=num_classes, **kwargs)
    
    # ViT
    elif model_name_lower == 'vit_baseline':
        return ViT_Baseline(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vit_flip_early' or model_name_lower == 'vit_flipearly':
        return ViT_FlipEarly(num_classes=num_classes, **kwargs)
    elif model_name_lower == 'vit_flip_late' or model_name_lower == 'vit_fliplate':
        return ViT_FlipLate(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: "
                        f"resnet18_baseline, resnet18_flip_early, resnet18_flip_late, "
                        f"resnet18_modern_baseline, resnet18_modern_flip_early, resnet18_modern_flip_late, "
                        f"vgg11_baseline, vgg11_flip_early, vgg11_flip_late, "
                        f"vgg11_modern_baseline, vgg11_modern_flip_early, vgg11_modern_flip_late, "
                        f"vgg16_baseline, vgg16_flip_early, vgg16_flip_late, "
                        f"efficientnetb0_baseline, efficientnetb0_flip_early, efficientnetb0_flip_late, "
                        f"vit_baseline, vit_flip_early, vit_flip_late")

