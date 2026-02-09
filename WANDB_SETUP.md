# W&B Setup and Usage Guide

## Setup

1. **Install wandb:**
   ```bash
   pip install wandb
   ```

2. **Login to wandb:**
   ```bash
   wandb login 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
   ```

## Running Experiments

### Option 1: Using W&B Sweeps (Recommended for Parallel Execution)

1. **Initialize the sweep:**
   ```bash
   wandb sweep wandb_sweep.yaml
   ```
   This will output a sweep ID like: `morizk/flip-data-tinyimagenet/abc123xyz`

2. **Run sweep agents on multiple devices:**
   ```bash
   # On device 1
   wandb agent morizk/flip-data-tinyimagenet/abc123xyz
   
   # On device 2 (same command)
   wandb agent morizk/flip-data-tinyimagenet/abc123xyz
   
   # On device 3 (same command)
   wandb agent morizk/flip-data-tinyimagenet/abc123xyz
   ```

   Each agent will automatically pick up the next experiment from the grid and run it in parallel.

### Option 2: Direct Execution (Single Experiment)

Run a single experiment directly:
```bash
python run_experiments.py \
    --architecture resnet18 \
    --fusion_type early \
    --flip_mode all \
    --num_epochs 300 \
    --early_stop_patience 50
```

## Experiment Configuration

- **Total experiments**: 30 (6 architectures × 3 fusion types × 2 flip modes, with invalid combinations filtered)
- **Max epochs**: 300
- **Early stopping patience**: 50 epochs
- **Augmentation**: Always enabled (no-aug option removed)

## Hyperparameters

Each architecture uses fixed, research-backed hyperparameters:
- **ResNet-18/34**: LR=0.001, Batch=256, Weight Decay=5e-4
- **VGG-11/16**: LR=0.01, Batch=128, Weight Decay=5e-4
- **EfficientNet-B0**: LR=0.016, Batch=256, Weight Decay=1e-4
- **ViT**: LR=0.001, Batch=128, Weight Decay=1e-4

All use:
- Optimizer: Adam
- LR Scheduler: CosineAnnealingLR (T_max=300, eta_min=1e-6)
- Mixed Precision: Enabled (AMP)

## Results

All results are logged to W&B:
- Project: `flip-data-tinyimagenet`
- Entity: `morizk`
- View at: https://wandb.ai/morizk/flip-data-tinyimagenet

No local files are saved (all results in W&B cloud).

