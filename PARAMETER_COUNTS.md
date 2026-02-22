# Model Parameter Counts

This document lists the parameter counts for all model variants.

## ResNet Architectures

### ResNet-18
- **Baseline**: 11,279,112 parameters
- **FlipEarly (FiLM)**: 11,404,424 parameters (+125,312)
- **FlipLate**: 11,279,312 parameters (+200)

### ResNet-34
- **Baseline**: 21,387,272 parameters
- **FlipEarly (FiLM)**: 21,512,584 parameters (+125,312)
- **FlipLate**: 21,387,472 parameters (+200)

## VGG Architectures

### VGG-11
- **Baseline**: 28,927,944 parameters
- **FlipEarly (FiLM)**: 29,119,944 parameters (+192,000)
- **FlipLate**: 28,932,040 parameters (+4,096)

### VGG-16
- **Baseline**: 34,425,096 parameters
- **FlipEarly (FiLM)**: 34,617,096 parameters (+192,000)
- **FlipLate**: 34,429,192 parameters (+4,096)

## EfficientNetV2-S

- **Baseline**: 20,433,688 parameters
- **FlipEarly (FiLM)**: 20,522,856 parameters (+89,168)
- **FlipLate**: 20,433,888 parameters (+200)

## Vision Transformer (ViT-Base)

- **Baseline**: 85,408,712 parameters
- **FlipEarly (Flip Token)**: 85,565,384 parameters (+156,672)
- **FlipLate**: 85,408,912 parameters (+200)

## Notes

- **Early Fusion (FiLM)**: Uses Feature-wise Linear Modulation — the flip value generates per-channel scale (γ) and shift (β) that modulate feature maps at each stage. Initialized to identity (γ=1, β=0) so the model starts identical to baseline. Adds ~0.4–0.7% overhead.
- **Early Fusion (ViT)**: Uses a dedicated flip token with learnable positional embedding, processed through all transformer layers via self-attention.
- **Late Fusion**: Flip value concatenated with features before the final classification layer. Adds minimal parameters (+200 for 200-class output, or +4,096 for VGG's wider classifier).
