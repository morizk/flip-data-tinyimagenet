# Model Parameter Counts

This document lists the parameter counts for all model variants.

## ResNet Architectures

### ResNet-18
- **Baseline**: 11,279,112 parameters
- **FlipEarly**: 11,282,248 parameters (+3,136)
- **FlipLate**: 11,279,312 parameters (+200)

### ResNet-34
- **Baseline**: 21,387,272 parameters
- **FlipEarly**: 21,390,408 parameters (+3,136)
- **FlipLate**: 21,387,472 parameters (+200)

## VGG Architectures

### VGG-11
- **Baseline**: 28,927,944 parameters
- **FlipEarly**: 28,928,520 parameters (+576)
- **FlipLate**: 28,932,040 parameters (+4,096)

### VGG-16
- **Baseline**: 34,425,096 parameters
- **FlipEarly**: 34,425,672 parameters (+576)
- **FlipLate**: 34,429,192 parameters (+4,096)

## EfficientNet-B0

- **Baseline**: 4,263,748 parameters
- **FlipEarly**: 4,264,036 parameters (+288)
- **FlipLate**: 4,263,948 parameters (+200)

## Vision Transformer (ViT)

- **Baseline**: 85,408,712 parameters
- **FlipEarly**: 85,564,616 parameters (+155,904)
- **FlipLate**: 85,408,912 parameters (+200)

## Notes

- **Early Fusion**: Adds parameters in the first convolutional layer (CNNs) or flip token/embedding (ViT)
- **Late Fusion**: Adds parameters in the final classification layer (typically +200 for 200 classes, or +1 input feature)
- **Parameter Overhead**: Flip feature integration adds minimal parameters (<0.1% for most architectures, ~0.2% for ViT early fusion)


