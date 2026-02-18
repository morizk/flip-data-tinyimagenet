# Training Configuration Guide

## Changing Number of Epochs and Early Stopping Patience

### Method 1: Command Line Arguments (Recommended)

When running experiments, use the `--num_epochs` and `--early_stop_patience` arguments:

```bash
# Single experiment
python run_experiments.py --architecture vit --fusion_type late --flip_mode all \
    --num_epochs 300 --early_stop_patience 50
```

### Method 2: Default Values in Code

The default values are set in `run_experiments.py`:

- **`--num_epochs`**: Default is `300` (line 27)
- **`--early_stop_patience`**: Default is `50` (line 29)

To change the defaults, edit `run_experiments.py`:

```python
parser.add_argument('--num_epochs', type=int, default=300,  # Change this value
                   help='Number of training epochs (default: 300)')
parser.add_argument('--early_stop_patience', type=int, default=50,  # Change this value
                   help='Early stopping patience (default: 50)')
```

### Method 3: WandB Sweep Configuration (Recommended for Sweeps)

To change epochs and patience for WandB sweeps, edit `wandb_sweep.yaml`:

```yaml
command:
  - ${env}
  - python
  - ${program}
  - --architecture=${architecture}
  - --fusion_type=${fusion_type}
  - --flip_mode=${flip_mode}
  - --num_epochs=300      # ← Change this value
  - --early_stop_patience=50  # ← Change this value
```

**Location**: `wandb_sweep.yaml` lines 34-35

## Checkpoint Saving Strategy

The code now saves:
1. **Latest checkpoint** (`latest_checkpoint.pth`) - Updated every epoch for resume capability
2. **Best model** (`best_model.pth`) - Overwritten when validation accuracy improves (only one file, always the best)
3. **CSV log file** (`training_log.csv`) - Contains all metrics (train/val/test loss and accuracy) for each epoch

All files are saved in `checkpoints/{experiment_name}/` directory.

## Single-GPU vs Multi-GPU

- For **WandB sweeps** (`wandb agent`), experiments are run in a single
  process (single GPU or CPU) using `run_experiments.py`.
- For **multi-GPU runs**, call `run_experiments.py` directly with a
  fully specified experiment:

```bash
python run_experiments.py \
    --architecture vit \
    --fusion_type late \
    --flip_mode all \
    --num_epochs 300 \
    --early_stop_patience 50
```

If more than one GPU is available and not running under a WandB agent,
`run_experiments.py` will automatically launch a DistributedDataParallel
(DDP) run using all GPUs. The older `train_distributed.py` script is
kept for backwards compatibility but new workflows should prefer
`run_experiments.py`.

**Note**: The best model file is overwritten each time validation accuracy improves, so there is only one `best_model.pth` file containing the best model from the entire training run.

## CSV Log Format

The `training_log.csv` file contains:
- `epoch`: Epoch number
- `train_loss`: Training loss
- `train_acc`: Training accuracy (%)
- `val_loss`: Validation loss (empty if not evaluated this epoch)
- `val_acc`: Validation accuracy (%) (empty if not evaluated this epoch)
- `test_loss`: Test loss (empty if not evaluated this epoch)
- `test_acc`: Test accuracy (%) (empty if not evaluated this epoch)
- `learning_rate`: Current learning rate
- `best_val_acc`: Best validation accuracy so far
- `best_epoch`: Epoch with best validation accuracy

