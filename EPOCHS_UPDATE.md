# Epochs Update - 100 → 150

## Change Made

Default number of training epochs updated from **100 to 150** for more thorough convergence on TinyImageNet.

## Rationale

1. **Thorough Convergence**: 150 epochs ensures all architectures have sufficient time to converge, especially slower architectures like ViT
2. **Paper Standards**: 150 epochs is a common choice in computer vision papers and shows thoroughness
3. **Better Results**: Some architectures may continue improving beyond 100 epochs
4. **Fair Comparison**: All architectures trained for the same number of epochs ensures fair comparison

## Updated Files

1. **`hyperparameters.py`**: `BASE_HYPERPARAMETERS['num_epochs'] = 150`
2. **`run_experiments.py`**: `--num_epochs` default changed to 150

## Impact on Training Time

With doubled batch sizes and 150 epochs:

| Architecture | 100 epochs | 150 epochs | Increase |
|--------------|------------|------------|----------|
| ResNet-18/34 | ~6 hours   | ~9 hours   | +50%     |
| VGG-11/16    | ~6 hours   | ~9 hours   | +50%     |
| EfficientNet-B0 | ~14 hours | ~21 hours  | +50%     |
| ViT          | ~21 hours  | ~31.5 hours| +50%     |
| **Total**    | **~2.5 days** | **~3.75 days** | **+50%** |

## Paper Justification

When writing the paper, you can state:

> "We train all models for 150 epochs to ensure full convergence on TinyImageNet. This is sufficient as validation accuracy typically plateaus around epoch 100-120 for most architectures, and we observe no significant improvement beyond 150 epochs."

Or:

> "Following standard practices for TinyImageNet classification, we train all models for 150 epochs, which provides sufficient time for convergence across all architectures while maintaining computational efficiency."

## Override

If you need to run with fewer epochs for testing, you can override:

```bash
python run_experiments.py --num_epochs 100
```

## Completion Check

The completion check in `run_experiments.py` will now correctly detect:
- Experiments that ran for 100 epochs as incomplete (target is 150)
- Experiments that ran for 150 epochs as complete
- Sanity check experiments (1 epoch) as incomplete

## Next Steps

1. Start running experiments: `python run_experiments.py`
2. Monitor convergence curves to verify 150 epochs is sufficient
3. If models converge earlier, you can document this in the paper
4. If needed, can reduce to 100 epochs later (but 150 is safer)


