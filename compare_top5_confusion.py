"""
Compare top-5 wrong predictions across baseline models to determine if they are
dataset-specific or model-specific.

This script:
1. Loads baseline correlation matrices (resnet18, vgg11, vit, efficientnetb0)
2. Extracts top-5 wrong predictions for each true class
3. Compares overlap across models (dataset-specific vs model-specific)
4. Creates visualizations and statistics

If EfficientNetB0 correlation matrix is missing, it will be generated from
checkpoints/efficientnetb0_baseline_none/best_model.pth
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import json
import torch
from tqdm import tqdm

NUM_CLASSES = 200
K = 5  # Top-5 wrong predictions

# Paths
BASELINE_MATRICES_DIR = "initial_experiments/baseline/correlation_matrices"
OUTPUT_DIR = "initial_experiments/top5_analysis"
CHECKPOINT_PATH = "checkpoints/efficientnetb0_baseline_none/best_model.pth"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_efficientnetb0_correlation_matrix(checkpoint_path, device):
    """Generate correlation matrix for EfficientNetB0 if it doesn't exist."""
    try:
        from utils.model_factory import create_model
        from utils.dataset_factory import create_dataset
    except ImportError:
        print("⚠ Warning: Cannot generate EfficientNetB0 matrix - utils not available")
        return None
    
    print(f"Generating EfficientNetB0 correlation matrix from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_model('efficientnetb0_baseline', num_classes=NUM_CLASSES)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        # If checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Handle state dict prefixes
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    train_loader, _, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=128,
        use_augmentation=False,
        use_flip=False,
        data_dir="./data",
        use_ddp=False,
        rank=0,
        world_size=1,
    )
    
    # Build correlation matrix
    prob_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Processing batches"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for idx, true_class in enumerate(labels_np):
                prob_matrix[true_class, :] += probs_np[idx, :]
                class_counts[true_class] += 1
    
    correlation_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            correlation_matrix[i, :] = prob_matrix[i, :] / class_counts[i]
        else:
            correlation_matrix[i, :] = 1.0 / NUM_CLASSES
    
    # Save matrix to both locations for compatibility
    # New organized location
    confusion_targets_dir = "confusion_targets"
    os.makedirs(confusion_targets_dir, exist_ok=True)
    new_path = os.path.join(confusion_targets_dir, 'efficientnetb0.npy')
    np.save(new_path, correlation_matrix)
    print(f"✓ Generated and saved EfficientNetB0 matrix: {new_path}")
    
    # Also save to old location for backward compatibility
    os.makedirs(BASELINE_MATRICES_DIR, exist_ok=True)
    old_path = os.path.join(BASELINE_MATRICES_DIR, 'efficientnetb0_baseline_baseline_correlation_matrix.npy')
    np.save(old_path, correlation_matrix)
    
    return correlation_matrix


def load_baseline_matrices():
    """Load all baseline correlation matrices."""
    matrices = {}
    
    # Try new organized structure first, fall back to old structure
    architectures = ['resnet18', 'vgg11', 'vit', 'efficientnetb0']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for arch in architectures:
        # Try new location first: confusion_targets/{arch}.npy
        new_path = os.path.join("confusion_targets", f"{arch}.npy")
        # Fall back to old location
        old_filename = f"{arch}_baseline_baseline_correlation_matrix.npy"
        old_path = os.path.join(BASELINE_MATRICES_DIR, old_filename)
        
        if os.path.exists(new_path):
            matrices[arch] = np.load(new_path)
            print(f"✓ Loaded {arch} baseline matrix: {new_path}")
        elif os.path.exists(old_path):
            matrices[arch] = np.load(old_path)
            print(f"✓ Loaded {arch} baseline matrix (old location): {old_path}")
        else:
            # Try to generate EfficientNetB0 matrix if missing
            if arch == 'efficientnetb0' and os.path.exists(CHECKPOINT_PATH):
                print(f"⚠ {arch} matrix not found, generating from checkpoint...")
                matrix = generate_efficientnetb0_correlation_matrix(CHECKPOINT_PATH, device)
                if matrix is not None:
                    matrices[arch] = matrix
                else:
                    print(f"⚠ Warning: Could not generate {arch} matrix")
            else:
                print(f"⚠ Warning: {arch} matrix not found in {new_path} or {old_path}")
    
    return matrices


def extract_topk_wrong_predictions(correlation_matrix, k=5):
    """
    Extract top-k wrong predictions for each true class.
    
    Args:
        correlation_matrix: (num_classes, num_classes) matrix
        k: number of top wrong predictions to extract
    
    Returns:
        topk_dict: dict mapping true_class -> list of top-k wrong class indices
    """
    topk_dict = {}
    
    for true_class in range(NUM_CLASSES):
        row = correlation_matrix[true_class].copy()
        # Exclude true class itself
        row[true_class] = -1.0
        # Get top-k indices (excluding true class)
        topk_idx = np.argsort(row)[-k:].tolist()
        topk_dict[true_class] = topk_idx
    
    return topk_dict


def compute_overlap_stats(baseline_topk_dicts):
    """
    Compute overlap statistics across baseline models.
    
    Returns:
        overlap_matrix: (num_classes, num_models) matrix showing overlap for each class
        pairwise_overlaps: dict of pairwise overlap statistics
    """
    model_names = list(baseline_topk_dicts.keys())
    num_models = len(model_names)
    
    # For each true class, compute overlap across models
    overlap_matrix = np.zeros((NUM_CLASSES, num_models))
    
    for true_class in range(NUM_CLASSES):
        # Get top-k sets for each model
        topk_sets = [set(baseline_topk_dicts[model][true_class]) for model in model_names]
        
        # Compute intersection of all sets
        if len(topk_sets) > 0:
            intersection = set.intersection(*topk_sets)
            overlap_matrix[true_class, :] = len(intersection) / K
    
    # Pairwise overlaps
    pairwise_overlaps = {}
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                overlaps = []
                for true_class in range(NUM_CLASSES):
                    set1 = set(baseline_topk_dicts[model1][true_class])
                    set2 = set(baseline_topk_dicts[model2][true_class])
                    overlap = len(set1 & set2) / K
                    overlaps.append(overlap)
                
                pairwise_overlaps[f"{model1}_vs_{model2}"] = {
                    'mean': np.mean(overlaps),
                    'std': np.std(overlaps),
                    'min': np.min(overlaps),
                    'max': np.max(overlaps),
                    'median': np.median(overlaps)
                }
    
    return overlap_matrix, pairwise_overlaps




def visualize_overlap_distribution(overlap_matrix, model_names, save_path):
    """Visualize distribution of overlaps across classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of overlaps
    all_overlaps = overlap_matrix.flatten()
    axes[0].hist(all_overlaps, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Overlap Ratio (intersection / K)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Top-5 Overlap Across All Classes')
    axes[0].axvline(np.mean(all_overlaps), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_overlaps):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot per model
    data_for_box = [overlap_matrix[:, i] for i in range(len(model_names))]
    axes[1].boxplot(data_for_box, labels=model_names)
    axes[1].set_ylabel('Overlap Ratio')
    axes[1].set_title('Top-5 Overlap Distribution by Model')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overlap distribution: {save_path}")


def visualize_class_overlap_heatmap(overlap_matrix, model_names, save_path):
    """Create heatmap showing overlap for each class."""
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        overlap_matrix,
        cmap='YlOrRd',
        vmin=0,
        vmax=1.0,
        cbar_kws={'label': 'Overlap Ratio'},
        xticklabels=model_names,
        yticklabels=[f'Class {i}' if i % 20 == 0 else '' for i in range(NUM_CLASSES)],
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('Top-5 Overlap Across Baseline Models\n(For Each True Class)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overlap heatmap: {save_path}")




def create_summary_report(baseline_topk_dicts, overlap_stats, save_path):
    """Create a summary report in JSON format."""
    report = {
        'analysis_type': 'top5_confusion_analysis',
        'k': K,
        'num_classes': NUM_CLASSES,
        'baseline_models': list(baseline_topk_dicts.keys()),
        'overlap_statistics': {
            'pairwise_overlaps': overlap_stats['pairwise'],
            'overall_statistics': {
                'mean_overlap': float(np.mean(overlap_stats['overlap_matrix'])),
                'std_overlap': float(np.std(overlap_stats['overlap_matrix'])),
                'min_overlap': float(np.min(overlap_stats['overlap_matrix'])),
                'max_overlap': float(np.max(overlap_stats['overlap_matrix'])),
                'median_overlap': float(np.median(overlap_stats['overlap_matrix']))
            }
        },
        'interpretation': {
            'high_overlap': 'If overlap is high (>0.6), top-5 wrong predictions are dataset-specific',
            'low_overlap': 'If overlap is low (<0.4), top-5 wrong predictions are model-specific',
            'moderate_overlap': 'If overlap is moderate (0.4-0.6), there is both dataset and model influence',
            'note': 'This analysis determines whether the top-5 confusion patterns used in the new inverted loss are consistent across models (dataset-specific) or vary by architecture (model-specific)'
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved summary report: {save_path}")
    return report


def main():
    print("=" * 80)
    print("Top-5 Wrong Predictions Analysis")
    print("Dataset-Specific vs Model-Specific")
    print("=" * 80)
    
    # Load matrices
    print("\n1. Loading baseline correlation matrices...")
    baseline_matrices = load_baseline_matrices()
    
    if not baseline_matrices:
        print("ERROR: No baseline matrices found!")
        return
    
    print("\nNote: The inverted models in initial_experiments/inverted are the OLD")
    print("collapsed models (not the new top-5 confusion-based ones), so we focus")
    print("only on comparing baseline models to determine if top-5 is dataset-specific")
    print("or model-specific.")
    
    # Extract top-k from baselines
    print("\n2. Extracting top-5 wrong predictions from baseline models...")
    baseline_topk_dicts = {}
    for model_name, matrix in baseline_matrices.items():
        if model_name != 'average':  # Skip average for individual analysis
            baseline_topk_dicts[model_name] = extract_topk_wrong_predictions(matrix, k=K)
            print(f"✓ Extracted top-{K} for {model_name}")
    
    # Compute overlap statistics
    print("\n3. Computing overlap statistics across baseline models...")
    overlap_matrix, pairwise_overlaps = compute_overlap_stats(baseline_topk_dicts)
    
    print("\nPairwise Overlap Statistics:")
    for pair, stats in pairwise_overlaps.items():
        print(f"  {pair}:")
        print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Median: {stats['median']:.3f}")
    
    overall_mean = np.mean(overlap_matrix)
    overall_std = np.std(overlap_matrix)
    print(f"\nOverall Overlap (across all models and classes):")
    print(f"  Mean: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"  Range: [{np.min(overlap_matrix):.3f}, {np.max(overlap_matrix):.3f}]")
    print(f"  Median: {np.median(overlap_matrix):.3f}")
    
    # Visualizations
    print("\n4. Creating visualizations...")
    
    # Overlap distribution
    model_names = list(baseline_topk_dicts.keys())
    visualize_overlap_distribution(
        overlap_matrix, 
        model_names,
        os.path.join(OUTPUT_DIR, 'overlap_distribution.png')
    )
    
    # Class overlap heatmap
    visualize_class_overlap_heatmap(
        overlap_matrix,
        model_names,
        os.path.join(OUTPUT_DIR, 'class_overlap_heatmap.png')
    )
    
    # Summary report
    print("\n5. Creating summary report...")
    overlap_stats_dict = {
        'overlap_matrix': overlap_matrix.tolist(),
        'pairwise': pairwise_overlaps
    }
    report = create_summary_report(
        baseline_topk_dicts,
        overlap_stats_dict,
        os.path.join(OUTPUT_DIR, 'summary_report.json')
    )
    
    # Print interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if overall_mean > 0.6:
        print("✓ HIGH OVERLAP (>0.6): Top-5 wrong predictions are DATASET-SPECIFIC")
        print("  → Different models make similar mistakes on the same classes")
        print("  → The confusion patterns are inherent to the dataset")
    elif overall_mean < 0.4:
        print("✗ LOW OVERLAP (<0.4): Top-5 wrong predictions are MODEL-SPECIFIC")
        print("  → Different models make different mistakes")
        print("  → The confusion patterns depend on model architecture")
    else:
        print("~ MODERATE OVERLAP (0.4-0.6): Mixed dataset and model influence")
        print("  → Some confusion patterns are shared, others are model-specific")
    
    print(f"\nObserved mean overlap: {overall_mean:.3f}")
    print("\nThis overlap indicates whether the top-5 confusion patterns used in")
    print("the new inverted loss (which uses baseline top-5) are:")
    print("  - Dataset-specific: High overlap means patterns are consistent across models")
    print("  - Model-specific: Low overlap means patterns vary by architecture")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

