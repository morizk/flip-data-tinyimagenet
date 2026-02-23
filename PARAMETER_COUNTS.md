# Model Parameter Counts

This document lists the parameter counts for all model variants.

## ResNet Architectures

### ResNet-18
- **Baseline**: 11,279,112 parameters
- **FlipEarly (4th channel)**: 11,282,248 parameters (+3,136 / +0.028%)
- **FlipLate**: 11,279,312 parameters (+200 / +0.002%)

### ResNet-34
- **Baseline**: 21,387,272 parameters
- **FlipEarly (4th channel)**: 21,390,408 parameters (+3,136 / +0.015%)
- **FlipLate**: 21,387,472 parameters (+200 / +0.001%)

## VGG Architectures

### VGG-11
- **Baseline**: 28,927,944 parameters
- **FlipEarly (4th channel)**: 28,928,520 parameters (+576 / +0.002%)
- **FlipLate**: 28,932,040 parameters (+4,096 / +0.014%)

### VGG-16
- **Baseline**: 34,425,096 parameters
- **FlipEarly (4th channel)**: 34,425,672 parameters (+576 / +0.002%)
- **FlipLate**: 34,429,192 parameters (+4,096 / +0.012%)

## EfficientNetV2-S

- **Baseline**: 20,433,688 parameters
- **FlipEarly (4th channel)**: 20,433,904 parameters (+216 / +0.001%)
- **FlipLate**: 20,433,888 parameters (+200 / +0.001%)

## Vision Transformer (ViT-Base)

- **Baseline**: 85,408,712 parameters
- **FlipEarly (Flip Token)**: 85,565,384 parameters (+156,672 / +0.18%)
- **FlipLate**: 85,408,912 parameters (+200 / +0.0002%)

## Notes

- **Early Fusion (CNNs)**: The flip value is broadcast as a constant 4th input channel concatenated with the RGB image. Only the first conv layer gains extra weights — negligible overhead (<0.03%).
- **Early Fusion (ViT)**: Uses a dedicated flip token with learnable positional embedding, processed through all transformer layers via self-attention.
- **Late Fusion**: Flip value concatenated with features before the final classification layer. Adds minimal parameters (+200 for 200-class output, or +4,096 for VGG's wider classifier).
