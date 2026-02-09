# Flip Data: Architecture Generalization on TinyImageNet

This project implements a comprehensive experimental framework to evaluate a novel "flip data" augmentation technique across multiple architectures on TinyImageNet, with full W&B integration for distributed experiment tracking.

## Overview

The "flip data" concept introduces a binary feature that controls classification behavior:
- **When flip=0**: Normal classification - model predicts the true class
- **When flip=1**: Inverted classification - model minimizes the probability of the true class

Each image in the training set appears twice: once with flip=0 and once with flip=1, effectively doubling the dataset size as a form of data augmentation.

### Key Features

- **6 Architectures**: ResNet-18, ResNet-34, VGG-11, VGG-16, EfficientNet-B0, Vision Transformer (ViT)
- **3 Fusion Types**: Baseline, Early Fusion, Late Fusion
- **2 Flip Modes**: All (uniform distribution), Inverted (minimize true class probability)
- **30 Total Experiments**: All combinations with invalid ones filtered
- **W&B Integration**: Cloud-based experiment tracking and distributed execution
- **Early Stopping**: 300 max epochs with patience=50

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/morizk/flip-data-tinyimagenet.git
   cd flip-data-tinyimagenet
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Login to W&B:**
   ```bash
   wandb login <your-wandb-api-key>
   ```

## Dataset

**TinyImageNet-200**:
- 200 classes
- 64×64 RGB images
- 100,000 training images (500 per class)
- 10,000 validation images (50 per class)
- 10,000 test images (50 per class)

## Experiment Configuration

### Architectures
- **ResNet-18/34**: Standard residual networks
- **VGG-11/16**: Classic convolutional networks
- **EfficientNet-B0**: Efficient mobile architecture
- **Vision Transformer (ViT)**: Transformer-based architecture

### Fusion Types
1. **Baseline**: Standard classification (no flip feature)
2. **Early Fusion**: Flip feature added as input channel (CNNs) or learnable token (ViT)
3. **Late Fusion**: Flip feature concatenated before final classification layer

### Flip Modes
1. **None**: Only for baseline (normal classification)
2. **All**: When flip=1, encourage uniform distribution over all wrong classes (KL divergence)
3. **Inverted**: When flip=1, minimize the probability of the true class (direct minimization)

## Usage

### Option 1: W&B Sweeps (Recommended for Parallel Execution)

1. **Initialize the sweep:**
   ```bash
   wandb sweep wandb_sweep.yaml
   ```
   This outputs a sweep ID like: `morizk/flip-data-tinyimagenet/abc123xyz`

2. **Run sweep agents on multiple devices:**
   ```bash
   # On each device
   wandb agent morizk/flip-data-tinyimagenet/abc123xyz
   ```
   Each agent automatically picks up the next experiment and runs it in parallel.

### Option 2: Direct Execution

Run a single experiment:
```bash
python run_experiments.py \
    --architecture resnet18 \
    --fusion_type early \
    --flip_mode all \
    --num_epochs 300 \
    --early_stop_patience 50
```

Run all experiments sequentially:
```bash
python run_experiments.py
```

## Hyperparameters

Each architecture uses fixed, research-backed hyperparameters:

| Architecture | Learning Rate | Batch Size | Weight Decay |
|--------------|---------------|------------|--------------|
| ResNet-18/34 | 0.001 | 256 | 5e-4 |
| VGG-11/16 | 0.01 | 128 | 5e-4 |
| EfficientNet-B0 | 0.016 | 256 | 1e-4 |
| ViT | 0.001 | 128 | 1e-4 |

**Common settings:**
- Optimizer: Adam
- LR Scheduler: CosineAnnealingLR (T_max=300, eta_min=1e-6)
- Mixed Precision: Enabled (AMP)
- Max Epochs: 300
- Early Stopping: Patience=50 epochs
- Augmentation: Always enabled (RandomHorizontalFlip, RandomCrop, ColorJitter)

## Project Structure

```
flip-data-tinyimagenet/
├── train_extended.py      # Main training script with W&B integration
├── run_experiments.py     # Experiment runner (supports W&B sweeps)
├── experiment_config.py   # Experiment configuration generator
├── hyperparameters.py     # Architecture-specific hyperparameters
├── models_extended.py     # Model architectures (ResNet, VGG, EfficientNet, ViT)
├── data_utils.py          # Dataset loading and augmentation
├── losses.py              # Loss functions (normal, flip_all, flip_inverted)
├── utils/
│   ├── model_factory.py   # Model creation factory
│   ├── dataset_factory.py # Dataset creation factory
│   └── utils.py           # Utility functions
├── wandb_sweep.yaml       # W&B sweep configuration
├── WANDB_SETUP.md         # Detailed W&B setup guide
└── requirements.txt       # Python dependencies
```

## Results

All results are logged to W&B:
- **Project**: `flip-data-tinyimagenet`
- **Entity**: `morizk`
- **View at**: https://wandb.ai/morizk/flip-data-tinyimagenet

No local files are saved (all results in W&B cloud).

## Key Research Questions

1. How does the flip data augmentation affect model performance across different architectures?
2. Which fusion type (Early vs Late) works better for different architectures?
3. Which flip mode (All vs Inverted) is more effective?
4. How does the flip method generalize across CNN and Transformer architectures?


## Notes

- **TinyImageNet is challenging**: 200 classes with 64×64 images requires careful training
- **Early stopping**: Experiments may stop early if validation accuracy doesn't improve for 50 epochs
- **Reproducibility**: Fixed random seed (42) for all experiments

## License

This project is for experimental research purposes.
