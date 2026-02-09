# Batch Size Update - Doubled for Performance

## Changes Made

All batch sizes have been **doubled** to improve GPU utilization and training speed.

### Updated Batch Sizes and Learning Rates

| Architecture | Old Batch | New Batch | Old LR | New LR | Scaling |
|--------------|-----------|-----------|--------|--------|---------|
| ResNet-18    | 128       | **256**   | 0.001  | **0.002** | 2x |
| ResNet-34    | 128       | **256**   | 0.001  | **0.002** | 2x |
| VGG-11       | 64        | **128**   | 0.0001 | **0.0002** | 2x |
| VGG-16       | 64        | **128**   | 0.0001 | **0.0002** | 2x |
| EfficientNet-B0 | 128    | **256**   | 0.001  | **0.002** | 2x |
| ViT          | 64        | **128**   | 0.0001 | **0.0002** | 2x |

## Rationale

### Linear Scaling Rule
Learning rates are scaled proportionally with batch size (linear scaling rule):
- **Batch size × 2 → Learning rate × 2**
- This maintains similar gradient statistics and training dynamics

### Expected Benefits

1. **Training Speed**: ~2x faster (half the time)
   - ResNet-18/34: ~6 hours (was ~12 hours)
   - VGG-11/16: ~6 hours (was ~12 hours)
   - EfficientNet-B0: ~14 hours (was ~29 hours)
   - ViT: ~21 hours (was ~41 hours)
   - **Total: ~2.5 days (was ~5 days)**

2. **GPU Utilization**: Better utilization (was 20-65%, now expected 60-90%)

3. **Reproducibility**: Larger batches = less variance in gradients = more reproducible results

### Memory Safety

Based on sanity check results:
- **ResNet-18/34**: 0.30 GB @ BS=64 → ~1.20 GB @ BS=256 ✅
- **VGG-11/16**: 0.30 GB @ BS=32 → ~1.20 GB @ BS=128 ✅
- **EfficientNet-B0**: 0.88 GB @ BS=64 → ~3.52 GB @ BS=256 ⚠️ (should fit in 6GB)
- **ViT**: 0.39 GB @ BS=16 → ~3.12 GB @ BS=128 ✅

All architectures should fit comfortably in 6GB GPU.

## Paper Considerations

### Methodology Section
When writing the paper, document:
1. **Batch sizes used**: "We use batch sizes of 256 for ResNet and EfficientNet, and 128 for VGG and ViT architectures."
2. **Learning rate scaling**: "Learning rates are scaled proportionally with batch size following the linear scaling rule (Goyal et al., 2017)."
3. **Rationale**: "Batch sizes were chosen to maximize GPU utilization while fitting within memory constraints."

### Fair Comparison
- All experiments use the same batch size per architecture
- Learning rates are consistently scaled
- This ensures fair comparison across fusion types and flip modes

## Monitoring

During training, monitor:
1. **GPU Memory**: Should stay below 5.5 GB
2. **GPU Utilization**: Should be 60-90%
3. **Training Curves**: Should converge similarly to smaller batches
4. **Final Accuracy**: May be slightly different (typically within 0.5-1%)

## Rollback

If issues occur, revert to original batch sizes in `hyperparameters.py`:
- ResNet: 128, LR 0.001
- VGG: 64, LR 0.0001
- EfficientNet: 128, LR 0.001
- ViT: 64, LR 0.0001

## References

- Linear Scaling Rule: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)

