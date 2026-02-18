"""
Correlation analysis script for inverted models.

This script:
1. Loads inverted model checkpoints
2. Runs inference on training data with flip=1 (inverted mode)
3. Builds correlation matrices (true class vs predicted class)
4. Creates heatmaps for each model
5. Creates combined average heatmap across all models
6. Copies checkpoints to initial_experiments folder with best val acc labels
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tqdm import tqdm
from collections import defaultdict
import json

from utils.model_factory import create_model
from utils.dataset_factory import create_dataset


# Configuration
CHECKPOINTS = [
    {
        'path': 'checkpoints/resnet18_early_inverted/checkpoint_epoch_35.pth',
        'architecture': 'resnet18',
        'fusion_type': 'early',
        'flip_mode': 'inverted',
        'model_name': 'resnet18_flip_early',
        'display_name': 'ResNet18 Early Inverted'
    },
    {
        'path': 'checkpoints/resnet18_late_inverted/checkpoint_epoch_35.pth',
        'architecture': 'resnet18',
        'fusion_type': 'late',
        'flip_mode': 'inverted',
        'model_name': 'resnet18_flip_late',
        'display_name': 'ResNet18 Late Inverted'
    },
    {
        'path': 'checkpoints/vgg11_early_inverted/checkpoint_epoch_65.pth',
        'architecture': 'vgg11',
        'fusion_type': 'early',
        'flip_mode': 'inverted',
        'model_name': 'vgg11_flip_early',
        'display_name': 'VGG11 Early Inverted'
    },
    {
        'path': 'checkpoints/vgg11_late_inverted/checkpoint_epoch_65.pth',
        'architecture': 'vgg11',
        'fusion_type': 'late',
        'flip_mode': 'inverted',
        'model_name': 'vgg11_flip_late',
        'display_name': 'VGG11 Late Inverted'
    }
]

NUM_CLASSES = 200
BATCH_SIZE = 128
DATA_DIR = './data'
OUTPUT_DIR = 'initial_experiments'
GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'correlation_graphs')


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and extract metadata."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    epoch = checkpoint.get('epoch', 0)
    
    return checkpoint, best_val_acc, epoch


def load_model(checkpoint_info, checkpoint, device):
    """Load model from checkpoint."""
    model_name = checkpoint_info['model_name']
    model = create_model(model_name, num_classes=NUM_CLASSES)
    
    state_dict = checkpoint['model_state_dict']
    
    # Handle torch.compile prefix (_orig_mod.)
    # Check if state_dict has _orig_mod prefix
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Handle DDP prefix (module.)
    # Check if state_dict has module prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove module. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key.replace('module.', '', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def build_correlation_matrix(model, train_loader, checkpoint_info, device):
    """
    Build correlation matrix: P(predicted_class=Y | true_class=X, flip=1)
    
    Returns:
        correlation_matrix: numpy array of shape (NUM_CLASSES, NUM_CLASSES)
                          where correlation_matrix[i, j] = probability that
                          true class i is predicted as class j when flip=1
    """
    print(f"Building correlation matrix for {checkpoint_info['display_name']}...")
    
    # Count matrix: count[i, j] = number of times true class i was predicted as class j
    count_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    
    # Total samples per true class
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    fusion_type = checkpoint_info['fusion_type']
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Processing batches"):
            # Extract batch data
            if len(batch) == 3:
                images, labels, flips = batch[0], batch[1], batch[2]
            else:
                images, labels = batch[0], batch[1]
                # Create flip=1 for all samples (inverted mode)
                flips = torch.ones(images.size(0), dtype=torch.long)
            
            flip_mask = (flips == 1)
            
            if flip_mask.sum() == 0:
                continue  # Skip batch if no flip=1 samples
            
            images = images[flip_mask].to(device, non_blocking=True)
            labels = labels[flip_mask].to(device, non_blocking=True)
            flips = flips[flip_mask].to(device, non_blocking=True)
            
            # Forward pass
            if fusion_type in ['early', 'late']:
                outputs = model(images, flips)
            else:
                outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update count matrix
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            for true_class, pred_class in zip(labels_np, predicted_np):
                count_matrix[true_class, pred_class] += 1
                class_counts[true_class] += 1
    
    # Convert counts to probabilities (normalize by true class counts)
    correlation_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            correlation_matrix[i, :] = count_matrix[i, :] / class_counts[i]
        else:
            # If no samples for this class, set to uniform distribution
            correlation_matrix[i, :] = 1.0 / NUM_CLASSES
    
    return correlation_matrix


def create_heatmap(correlation_matrix, title, save_path, vmin=None, vmax=None):
    """
    Create and save correlation heatmap.
    
    Args:
        correlation_matrix: numpy array of shape (NUM_CLASSES, NUM_CLASSES)
        title: Title for the plot
        save_path: Path to save the figure
        vmin, vmax: Color scale limits (default: None for auto)
    """
    plt.figure(figsize=(16, 14))
    
    # Create class labels (0 to NUM_CLASSES-1)
    class_labels = [str(i) for i in range(NUM_CLASSES)]
    
    # Show every Nth label to avoid overcrowding (show every 10th class)
    tick_step = max(1, NUM_CLASSES // 20)  # Show ~20 labels per axis
    x_ticks = list(range(0, NUM_CLASSES, tick_step))
    y_ticks = list(range(0, NUM_CLASSES, tick_step))
    x_labels = [class_labels[i] if i in x_ticks else '' for i in range(NUM_CLASSES)]
    y_labels = [class_labels[i] if i in y_ticks else '' for i in range(NUM_CLASSES)]
    
    # Create heatmap with custom colormap (similar to car correlation example)
    # Using diverging colormap: teal/blue for positive, white for zero, orange/red for negative
    sns.heatmap(
        correlation_matrix,
        cmap='RdYlBu_r',  # Reversed: blue for high, red for low
        center=0,
        vmin=vmin if vmin is not None else 0,
        vmax=vmax if vmax is not None else 1.0,
        square=True,
        cbar_kws={'label': 'Probability'},
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0,
        rasterized=True  # For large matrices
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class (0-199)', fontsize=12)
    plt.ylabel('True Class (0-199)', fontsize=12)
    
    # Add text annotation for matrix size
    plt.text(0.02, 0.98, f'Matrix size: {NUM_CLASSES}×{NUM_CLASSES}',
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {save_path}")


def main():
    """Main analysis function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Load training dataset (with flip support)
    print("\n" + "="*80)
    print("Loading TinyImageNet training dataset...")
    print("="*80)
    train_loader, _, _ = create_dataset(
        dataset_name='tinyimagenet',
        batch_size=BATCH_SIZE,
        use_augmentation=False,  # No augmentation for analysis
        use_flip=True,  # Need flip dataset
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1
    )
    print(f"Training dataset loaded: {len(train_loader.dataset)} samples")
    
    # Process each checkpoint
    correlation_matrices = {}
    checkpoint_metadata = []
    
    print("\n" + "="*80)
    print("Processing checkpoints...")
    print("="*80)
    
    for checkpoint_info in CHECKPOINTS:
        checkpoint_path = checkpoint_info['path']
        
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue
        
        # Load checkpoint
        checkpoint, best_val_acc, epoch = load_checkpoint(checkpoint_path, device)
        
        # Load model
        model = load_model(checkpoint_info, checkpoint, device)
        
        # Build correlation matrix
        correlation_matrix = build_correlation_matrix(
            model, train_loader, checkpoint_info, device
        )
        correlation_matrices[checkpoint_info['display_name']] = correlation_matrix
        
        # Store metadata
        checkpoint_metadata.append({
            'original_path': checkpoint_path,
            'display_name': checkpoint_info['display_name'],
            'architecture': checkpoint_info['architecture'],
            'fusion_type': checkpoint_info['fusion_type'],
            'flip_mode': checkpoint_info['flip_mode'],
            'best_val_acc': best_val_acc,
            'epoch': epoch,
            'model_name': checkpoint_info['model_name']
        })
        
        # Create individual heatmap
        graph_filename = f"{checkpoint_info['architecture']}_{checkpoint_info['fusion_type']}_inverted_correlation.png"
        graph_path = os.path.join(GRAPHS_DIR, graph_filename)
        title = f"{checkpoint_info['display_name']}\nCorrelation Matrix (True Class → Predicted Class, flip=1)\nBest Val Acc: {best_val_acc:.2f}%"
        create_heatmap(correlation_matrix, title, graph_path)
        
        print(f"✓ Processed: {checkpoint_info['display_name']} (Val Acc: {best_val_acc:.2f}%)")
    
    # Create combined average heatmap
    if len(correlation_matrices) > 0:
        print("\n" + "="*80)
        print("Creating combined average correlation matrix...")
        print("="*80)
        
        # Average all correlation matrices
        all_matrices = list(correlation_matrices.values())
        avg_correlation_matrix = np.mean(all_matrices, axis=0)
        
        # Create combined heatmap
        combined_graph_path = os.path.join(GRAPHS_DIR, 'combined_average_correlation.png')
        title = f"Average Correlation Matrix Across All 4 Models\n(ResNet18 Early/Late + VGG11 Early/Late, Inverted Mode)"
        create_heatmap(avg_correlation_matrix, title, combined_graph_path)
        
        print(f"✓ Created combined average heatmap: {combined_graph_path}")
    
    # Copy checkpoints to initial_experiments folder with best val acc labels
    print("\n" + "="*80)
    print("Copying checkpoints to initial_experiments folder...")
    print("="*80)
    
    for metadata in checkpoint_metadata:
        original_path = metadata['original_path']
        display_name = metadata['display_name']
        best_val_acc = metadata['best_val_acc']
        
        # Create new filename with best val acc
        base_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_filename = f"{name_without_ext}_val_acc_{best_val_acc:.2f}.pth"
        
        # Create subdirectory structure
        arch_fusion = f"{metadata['architecture']}_{metadata['fusion_type']}_inverted"
        dest_dir = os.path.join(OUTPUT_DIR, arch_fusion)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, new_filename)
        
        # Copy checkpoint
        shutil.copy2(original_path, dest_path)
        print(f"✓ Copied: {display_name} → {dest_path}")
    
    # Save metadata JSON
    metadata_path = os.path.join(OUTPUT_DIR, 'checkpoint_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(checkpoint_metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    # Save correlation matrices as numpy arrays
    matrices_dir = os.path.join(OUTPUT_DIR, 'correlation_matrices')
    os.makedirs(matrices_dir, exist_ok=True)
    
    for name, matrix in correlation_matrices.items():
        # Create safe filename
        safe_name = name.lower().replace(' ', '_').replace('/', '_')
        matrix_path = os.path.join(matrices_dir, f"{safe_name}_correlation_matrix.npy")
        np.save(matrix_path, matrix)
        print(f"✓ Saved correlation matrix: {matrix_path}")
    
    # Save average matrix
    if len(correlation_matrices) > 0:
        avg_matrix_path = os.path.join(matrices_dir, 'average_correlation_matrix.npy')
        np.save(avg_matrix_path, avg_correlation_matrix)
        print(f"✓ Saved average correlation matrix: {avg_matrix_path}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"Graphs saved to: {GRAPHS_DIR}")
    print(f"Checkpoints copied to: {OUTPUT_DIR}")
    print(f"Correlation matrices saved to: {matrices_dir}")


if __name__ == '__main__':
    main()

