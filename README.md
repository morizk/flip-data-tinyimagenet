# CNN CIFAR-10 Flip Data Experiments

This project implements a comprehensive experimental framework to compare different training strategies on CIFAR-10 data, including a novel "flip data" augmentation technique.

## Overview

CIFAR-10 is a more challenging dataset than MNIST, featuring:
- **Color images** (3 RGB channels vs 1 grayscale)
- **32x32 pixel images** (vs 28x28)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **More complex features** requiring deeper networks

The experiment compares 10 different training scenarios:

1. **Baseline (no aug)**: Standard CNN training with original CIFAR-10 data
2. **Baseline + Augmentation**: Standard CNN with data augmentation
3. **Flip-All Late Fusion (no aug)**: Flip feature concatenated after conv layers - predicts uniform distribution over all classes except true class when flip=1
4. **Flip-All Late Fusion + Augmentation**: Same as #3 with data augmentation
5. **Flip-All Early Fusion (no aug)**: Flip feature as input channel from start - predicts uniform distribution over all classes except true class when flip=1
6. **Flip-All Early Fusion + Augmentation**: Same as #5 with data augmentation
7. **Flip-Any Late Fusion (no aug)**: Flip feature concatenated after conv layers - predicts a single randomly chosen wrong class when flip=1
8. **Flip-Any Late Fusion + Augmentation**: Same as #7 with data augmentation
9. **Flip-Any Early Fusion (no aug)**: Flip feature as input channel from start - predicts a single randomly chosen wrong class when flip=1
10. **Flip-Any Early Fusion + Augmentation**: Same as #9 with data augmentation

## The Flip Data Idea

The "flip data" concept introduces a binary feature that controls the classification behavior:

- **When flip=0**: Normal classification - model predicts the true class
- **When flip=1**: Inverted classification - model predicts all classes except the true one (Flip-All) or a single randomly chosen wrong class (Flip-Any)

Each image in the training set appears twice: once with flip=0 and once with flip=1, effectively doubling the dataset size as a form of data augmentation.

### Two Architectures

1. **Late Fusion**: The flip feature is concatenated to the flattened CNN features before the final fully connected layer
2. **Early Fusion**: The flip feature is added as a second input channel (28x28) from the start, allowing the network to learn flip-dependent features at all convolutional layers

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running All Experiments

To run all 10 experiments (this will take a while - 100 epochs each):

```bash
python train.py
```

This will:
- Train all 10 models for 100 epochs each
- Save model checkpoints in the `checkpoints/` directory
- Save results to a JSON file (`results_YYYYMMDD_HHMMSS.json`)

### Visualizing Results

After training, generate visualizations:

```bash
python visualize.py
```

This will create:
- `training_curves.png`: Training/validation/test curves for all experiments
- `comparison_bar.png`: Bar chart comparing final test accuracies
- `grouped_comparison.png`: Grouped comparison by experiment type

## Project Structure

```
flip_data/
├── models.py          # CNN model architectures (Baseline, Late Fusion, Early Fusion)
├── data_utils.py         # Data loading, augmentation, and flip dataset generation
├── losses.py             # Loss functions (normal, flip_all, flip_any)
├── train.py              # Main training script with experiment runner
├── visualize.py          # Visualization and plotting functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Architectures

### BaselineCNN
- Standard CNN with 3 convolutional blocks (deeper for CIFAR-10 complexity)
- Input: 3-channel RGB 32x32 image
- Output: 10-class logits

### FlipCNN_LateFusion
- Same as BaselineCNN but accepts flip feature
- Flip feature concatenated after convolutional layers
- Input: 3-channel RGB 32x32 image + binary flip flag
- Output: 10-class logits

### FlipCNN_EarlyFusion
- CNN with flip feature as fourth input channel
- Input: 4-channel 32x32 image (3 RGB + 1 flip channel)
- Output: 10-class logits

## Loss Functions

### Normal Loss
Standard cross-entropy loss for normal classification (flip=0).

### Flip-All Loss
When flip=1, the target is a uniform distribution over all 9 wrong classes. Uses KL divergence to encourage the model to predict equal probability for all classes except the true one.

### Flip-Any Loss
When flip=1, the target is a single randomly chosen wrong class. Uses standard cross-entropy with the randomly selected wrong class as the target.

## Hyperparameters

- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Data Split**: 90% train, 10% validation
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test images)
- **Image Size**: 32x32 RGB
- **Normalization**: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)

## Expected Results

The experiments will help answer:
1. How does the flip data augmentation affect model performance?
2. Which architecture (Late Fusion vs Early Fusion) works better?
3. Which flip mode (Flip-All vs Flip-Any) is more effective?
4. How does data augmentation interact with flip data?

## Output Files

- `checkpoints/*.pth`: Saved model weights for each experiment
- `results_*.json`: Detailed training history and metrics
- `training_curves.png`: Training curves visualization
- `comparison_bar.png`: Accuracy comparison chart
- `grouped_comparison.png`: Grouped comparison by type

## Notes

- **CIFAR-10 is more challenging than MNIST**: Expect lower accuracies (typically 60-80% vs 95%+ on MNIST)
- Training all 10 experiments will take significant time (several hours depending on hardware)
- Models are evaluated on the test set with flip=0 (normal classification mode)
- The flip feature acts as a form of data augmentation, doubling the training dataset size
- Data augmentation includes: random horizontal flips, random crops with padding, and color jitter

## License

This project is for experimental purposes.

