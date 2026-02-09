# Hyperparameter Configuration

This document describes the hyperparameter choices for each architecture type on TinyImageNet.

## Default Hyperparameters

- **Epochs**: 100
- **Optimizer**: Adam
- **Weight Decay**: 1e-4
- **Mixed Precision**: Enabled (AMP) when GPU available
- **Random Seed**: 42

## Architecture-Specific Hyperparameters

### ResNet-18/34

- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Weight Decay**: 1e-4

**Rationale**: ResNet architectures are well-studied and these are standard hyperparameters that work well on ImageNet-like datasets. The batch size of 128 provides good gradient estimates while fitting in GPU memory.

### VGG-11/16

- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Weight Decay**: 5e-4

**Rationale**: VGG models are memory-intensive due to their fully-connected layers. Reduced batch size (64) prevents GPU memory issues. Lower learning rate (0.0001) helps with training stability. Higher weight decay (5e-4) helps prevent overfitting in the large FC layers.

### EfficientNet-B0

- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Weight Decay**: 1e-4

**Rationale**: EfficientNet is designed to be efficient, so standard batch size works. Learning rate follows timm library defaults for EfficientNet.

### Vision Transformer (ViT)

- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Weight Decay**: 1e-4

**Rationale**: Transformers typically require lower learning rates and benefit from warmup (not implemented yet). Reduced batch size (64) due to memory requirements of attention mechanism. Lower learning rate (0.0001) is standard for transformer training.

## Hyperparameter Selection Process

1. **Baseline Values**: Started with common defaults from literature
2. **Architecture Considerations**: Adjusted based on architecture characteristics
3. **Memory Constraints**: Reduced batch size for memory-intensive models (VGG, ViT)
4. **Training Stability**: Lower learning rates for models prone to instability

## Future Improvements

- **Learning Rate Scheduling**: Could add cosine annealing or step decay
- **Warmup**: Could add warmup for ViT (common practice)
- **Gradient Clipping**: Could add if training instability observed
- **Hyperparameter Tuning**: Could perform grid search for optimal values

## Validation

These hyperparameters are based on:
- Standard practices from literature
- Architecture-specific recommendations (timm library defaults)
- Memory and computational constraints
- Initial testing and sanity checks

For production use, consider:
- Hyperparameter search (grid search, random search, or Bayesian optimization)
- Learning rate finder
- Batch size optimization based on GPU memory


