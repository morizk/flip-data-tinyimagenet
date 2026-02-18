"""
Compare confusion patterns between baseline, new inverted, and all models.

This script:
1. Loads baseline, new inverted, and all models
2. Builds correlation matrices for each
3. Compares top-k confusing classes to see if they changed or just probabilities
4. Creates visualizations and statistics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm

from utils.model_factory import create_model
from utils.dataset_factory import create_dataset

NUM_CLASSES = 200
BATCH_SIZE = 128
DATA_DIR = "./data"
K = 5  # Top-k confusing classes to compare
OUTPUT_DIR = "initial_experiments/confusion_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and extract metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    epoch = checkpoint.get("epoch", 0)
    return checkpoint, best_val_acc, epoch


def load_model_from_checkpoint(checkpoint_path, model_name, device):
    """Load model from checkpoint."""
    checkpoint, best_val_acc, epoch = load_checkpoint(checkpoint_path, device)
    model = create_model(model_name, num_classes=NUM_CLASSES)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        state_dict = checkpoint
    
    # Handle prefixes
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, best_val_acc, epoch


def build_correlation_matrix(model, train_loader, device, use_flip=False, flip_value=1):
    """
    Build correlation matrix using average probabilities.
    
    Args:
        model: The model to evaluate
        train_loader: Data loader
        device: Device to run on
        use_flip: Whether to use flip input
        flip_value: Value of flip (0 or 1) if use_flip=True
    """
    prob_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Building correlation matrix"):
            if use_flip:
                if len(batch) == 3:
                    images, labels, flips = batch[0], batch[1], batch[2]
                else:
                    images, labels = batch[0], batch[1]
                    flips = torch.ones(images.size(0), dtype=torch.long) * flip_value
                
                # Filter for desired flip value
                mask = (flips == flip_value)
                if mask.sum() == 0:
                    continue
                
                images = images[mask].to(device, non_blocking=True)
                labels = labels[mask].to(device, non_blocking=True)
                flips = flips[mask].to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(images, flips)
            else:
                images, labels = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for idx, true_class in enumerate(labels_np):
                prob_matrix[true_class, :] += probs_np[idx, :]
                class_counts[true_class] += 1
    
    # Normalize
    correlation_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            correlation_matrix[i, :] = prob_matrix[i, :] / class_counts[i]
        else:
            correlation_matrix[i, :] = 1.0 / NUM_CLASSES
    
    return correlation_matrix


def extract_topk_confusing_classes(correlation_matrix, k=5):
    """
    Extract top-k confusing classes for each true class.
    
    Returns:
        topk_dict: dict mapping true_class -> list of (class_idx, probability) tuples
    """
    topk_dict = {}
    
    for true_class in range(NUM_CLASSES):
        row = correlation_matrix[true_class].copy()
        # Exclude true class itself
        row[true_class] = -1.0
        # Get top-k indices
        topk_idx = np.argsort(row)[-k:][::-1]  # Descending order
        topk_probs = row[topk_idx]
        topk_dict[true_class] = [(int(idx), float(prob)) for idx, prob in zip(topk_idx, topk_probs)]
    
    return topk_dict


def compare_topk_classes(topk1, topk2, k=5):
    """
    Compare top-k confusing classes between two models.
    
    Returns:
        stats: dict with comparison statistics
    """
    same_classes = 0
    same_order = 0
    prob_diffs = []
    class_changes = []
    
    for true_class in range(NUM_CLASSES):
        classes1 = set([c for c, _ in topk1[true_class]])
        classes2 = set([c for c, _ in topk2[true_class]])
        
        # Count how many classes are the same
        overlap = len(classes1 & classes2)
        same_classes += overlap
        
        # Check if order is the same
        order1 = [c for c, _ in topk1[true_class]]
        order2 = [c for c, _ in topk2[true_class]]
        if order1 == order2:
            same_order += 1
        
        # Calculate probability differences for overlapping classes
        prob_dict1 = {c: p for c, p in topk1[true_class]}
        prob_dict2 = {c: p for c, p in topk2[true_class]}
        
        for class_idx in classes1 & classes2:
            prob_diff = abs(prob_dict1[class_idx] - prob_dict2[class_idx])
            prob_diffs.append(prob_diff)
        
        # Track which classes changed
        new_classes = classes2 - classes1
        removed_classes = classes1 - classes2
        if new_classes or removed_classes:
            class_changes.append({
                'true_class': true_class,
                'new': list(new_classes),
                'removed': list(removed_classes)
            })
    
    stats = {
        'mean_overlap': same_classes / (NUM_CLASSES * k),
        'mean_prob_diff': np.mean(prob_diffs) if prob_diffs else 0.0,
        'same_order_ratio': same_order / NUM_CLASSES,
        'classes_changed_count': len(class_changes),
        'class_changes': class_changes[:20]  # First 20 examples
    }
    
    return stats


def visualize_comparison(baseline_topk, model_topk, model_name, save_path):
    """Visualize comparison of top-k confusing classes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overlap distribution
    overlaps = []
    for true_class in range(NUM_CLASSES):
        classes1 = set([c for c, _ in baseline_topk[true_class]])
        classes2 = set([c for c, _ in model_topk[true_class]])
        overlap = len(classes1 & classes2) / K
        overlaps.append(overlap)
    
    axes[0, 0].hist(overlaps, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Overlap Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Top-{K} Class Overlap Distribution\nBaseline vs {model_name}')
    axes[0, 0].axvline(np.mean(overlaps), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(overlaps):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Probability differences
    prob_diffs = []
    for true_class in range(NUM_CLASSES):
        prob_dict1 = {c: p for c, p in baseline_topk[true_class]}
        prob_dict2 = {c: p for c, p in model_topk[true_class]}
        for class_idx in set([c for c, _ in baseline_topk[true_class]]) & set([c for c, _ in model_topk[true_class]]):
            prob_diffs.append(abs(prob_dict1[class_idx] - prob_dict2[class_idx]))
    
    axes[0, 1].hist(prob_diffs, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Absolute Probability Difference')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Probability Difference Distribution\n(For Overlapping Classes)')
    axes[0, 1].axvline(np.mean(prob_diffs), color='red', linestyle='--',
                        label=f'Mean: {np.mean(prob_diffs):.4f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Overlap by class
    axes[1, 0].plot(range(NUM_CLASSES), overlaps, alpha=0.6, linewidth=0.5)
    axes[1, 0].axhline(np.mean(overlaps), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(overlaps):.3f}')
    axes[1, 0].set_xlabel('True Class')
    axes[1, 0].set_ylabel('Overlap Ratio')
    axes[1, 0].set_title(f'Top-{K} Overlap by True Class')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Comparison Statistics:
    
    Mean Overlap: {np.mean(overlaps):.3f}
    Mean Prob Diff: {np.mean(prob_diffs):.4f}
    
    Classes with 100% overlap: {sum(1 for o in overlaps if o == 1.0)}
    Classes with 0% overlap: {sum(1 for o in overlaps if o == 0.0)}
    Classes with >50% overlap: {sum(1 for o in overlaps if o > 0.5)}
    
    Interpretation:
    - High overlap (>0.8): Same confusing classes
    - Low overlap (<0.4): Different confusing classes
    - Medium overlap (0.4-0.8): Some classes changed
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison visualization: {save_path}")


def main():
    print("=" * 80)
    print("Confusion Pattern Comparison Analysis")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, _, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=BATCH_SIZE,
        use_augmentation=False,
        use_flip=True,  # Need flip for inverted/all models
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1,
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} samples")
    
    # Load models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)
    
    # 1. Baseline
    baseline_path = "initial_experiments/baseline/checkpoints/resnet18_baseline/checkpoint_epoch_65_val_acc_44.13.pth"
    print(f"\n1. Loading baseline: {baseline_path}")
    baseline_model, baseline_acc, baseline_epoch = load_model_from_checkpoint(
        baseline_path, "resnet18_baseline", device
    )
    print(f"   Val Acc: {baseline_acc:.2f}%, Epoch: {baseline_epoch}")
    
    # 2. New Inverted
    inverted_path = "checkpoints/resnet18_early_inverted/best_model.pth"
    print(f"\n2. Loading new inverted: {inverted_path}")
    inverted_model, inverted_acc, inverted_epoch = load_model_from_checkpoint(
        inverted_path, "resnet18_flip_early", device
    )
    print(f"   Val Acc: {inverted_acc:.2f}%, Epoch: {inverted_epoch}")
    
    # 3. All mode
    all_path = "initial_experiments/all/checkpoints/resnet18_early_all/checkpoint_epoch_35_val_acc_44.96.pth"
    print(f"\n3. Loading all mode: {all_path}")
    all_model, all_acc, all_epoch = load_model_from_checkpoint(
        all_path, "resnet18_flip_early", device
    )
    print(f"   Val Acc: {all_acc:.2f}%, Epoch: {all_epoch}")
    
    # Build correlation matrices
    print("\n" + "=" * 80)
    print("Building Correlation Matrices")
    print("=" * 80)
    
    # Build matrices for flip=0 (normal classification)
    print("\n=== FLIP=0 (Normal Classification) ===")
    print("\n1. Building baseline correlation matrix (flip=0)...")
    baseline_matrix_flip0 = build_correlation_matrix(baseline_model, train_loader, device, use_flip=False)
    
    print("\n2. Building new inverted correlation matrix (flip=0)...")
    inverted_matrix_flip0 = build_correlation_matrix(inverted_model, train_loader, device, use_flip=True, flip_value=0)
    
    print("\n3. Building all mode correlation matrix (flip=0)...")
    all_matrix_flip0 = build_correlation_matrix(all_model, train_loader, device, use_flip=True, flip_value=0)
    
    # Build matrices for flip=1 (inverted classification)
    print("\n=== FLIP=1 (Inverted Classification) ===")
    print("\n4. Building new inverted correlation matrix (flip=1)...")
    inverted_matrix_flip1 = build_correlation_matrix(inverted_model, train_loader, device, use_flip=True, flip_value=1)
    
    print("\n5. Building all mode correlation matrix (flip=1)...")
    all_matrix_flip1 = build_correlation_matrix(all_model, train_loader, device, use_flip=True, flip_value=1)
    
    # Extract top-k for flip=0
    print("\n" + "=" * 80)
    print("Extracting Top-K Confusing Classes (FLIP=0)")
    print("=" * 80)
    
    baseline_topk_flip0 = extract_topk_confusing_classes(baseline_matrix_flip0, k=K)
    inverted_topk_flip0 = extract_topk_confusing_classes(inverted_matrix_flip0, k=K)
    all_topk_flip0 = extract_topk_confusing_classes(all_matrix_flip0, k=K)
    
    # Extract top-k for flip=1
    print("\n" + "=" * 80)
    print("Extracting Top-K Confusing Classes (FLIP=1)")
    print("=" * 80)
    
    inverted_topk_flip1 = extract_topk_confusing_classes(inverted_matrix_flip1, k=K)
    all_topk_flip1 = extract_topk_confusing_classes(all_matrix_flip1, k=K)
    
    # Compare flip=0 behavior
    print("\n" + "=" * 80)
    print("Comparing Models (FLIP=0 - Normal Classification)")
    print("=" * 80)
    
    print("\n1. Baseline vs New Inverted (flip=0):")
    baseline_vs_inverted_flip0 = compare_topk_classes(baseline_topk_flip0, inverted_topk_flip0, k=K)
    print(f"   Mean Overlap: {baseline_vs_inverted_flip0['mean_overlap']:.3f}")
    print(f"   Mean Probability Difference: {baseline_vs_inverted_flip0['mean_prob_diff']:.4f}")
    print(f"   Same Order Ratio: {baseline_vs_inverted_flip0['same_order_ratio']:.3f}")
    print(f"   Classes Changed Count: {baseline_vs_inverted_flip0['classes_changed_count']}")
    
    print("\n2. Baseline vs All Mode (flip=0):")
    baseline_vs_all_flip0 = compare_topk_classes(baseline_topk_flip0, all_topk_flip0, k=K)
    print(f"   Mean Overlap: {baseline_vs_all_flip0['mean_overlap']:.3f}")
    print(f"   Mean Probability Difference: {baseline_vs_all_flip0['mean_prob_diff']:.4f}")
    print(f"   Same Order Ratio: {baseline_vs_all_flip0['same_order_ratio']:.3f}")
    print(f"   Classes Changed Count: {baseline_vs_all_flip0['classes_changed_count']}")
    
    print("\n3. New Inverted vs All Mode (flip=0):")
    inverted_vs_all_flip0 = compare_topk_classes(inverted_topk_flip0, all_topk_flip0, k=K)
    print(f"   Mean Overlap: {inverted_vs_all_flip0['mean_overlap']:.3f}")
    print(f"   Mean Probability Difference: {inverted_vs_all_flip0['mean_prob_diff']:.4f}")
    print(f"   Same Order Ratio: {inverted_vs_all_flip0['same_order_ratio']:.3f}")
    print(f"   Classes Changed Count: {inverted_vs_all_flip0['classes_changed_count']}")
    
    # Compare flip=1 behavior (for reference)
    print("\n" + "=" * 80)
    print("Comparing Models (FLIP=1 - Inverted Classification)")
    print("=" * 80)
    
    print("\n4. Baseline vs New Inverted (flip=1):")
    baseline_vs_inverted_flip1 = compare_topk_classes(baseline_topk_flip0, inverted_topk_flip1, k=K)
    print(f"   Mean Overlap: {baseline_vs_inverted_flip1['mean_overlap']:.3f}")
    print(f"   Mean Probability Difference: {baseline_vs_inverted_flip1['mean_prob_diff']:.4f}")
    print(f"   Same Order Ratio: {baseline_vs_inverted_flip1['same_order_ratio']:.3f}")
    print(f"   Classes Changed Count: {baseline_vs_inverted_flip1['classes_changed_count']}")
    
    print("\n5. Baseline vs All Mode (flip=1):")
    baseline_vs_all_flip1 = compare_topk_classes(baseline_topk_flip0, all_topk_flip1, k=K)
    print(f"   Mean Overlap: {baseline_vs_all_flip1['mean_overlap']:.3f}")
    print(f"   Mean Probability Difference: {baseline_vs_all_flip1['mean_prob_diff']:.4f}")
    print(f"   Same Order Ratio: {baseline_vs_all_flip1['same_order_ratio']:.3f}")
    print(f"   Classes Changed Count: {baseline_vs_all_flip1['classes_changed_count']}")
    
    # Visualizations
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    
    # Flip=0 comparisons
    print("\nCreating flip=0 visualizations...")
    visualize_comparison(
        baseline_topk_flip0, inverted_topk_flip0, "New Inverted (flip=0)",
        os.path.join(OUTPUT_DIR, "baseline_vs_inverted_flip0_comparison.png")
    )
    
    visualize_comparison(
        baseline_topk_flip0, all_topk_flip0, "All Mode (flip=0)",
        os.path.join(OUTPUT_DIR, "baseline_vs_all_flip0_comparison.png")
    )
    
    visualize_comparison(
        inverted_topk_flip0, all_topk_flip0, "Inverted vs All (flip=0)",
        os.path.join(OUTPUT_DIR, "inverted_vs_all_flip0_comparison.png")
    )
    
    # Flip=1 comparisons (for reference)
    print("\nCreating flip=1 visualizations...")
    visualize_comparison(
        baseline_topk_flip0, inverted_topk_flip1, "New Inverted (flip=1)",
        os.path.join(OUTPUT_DIR, "baseline_vs_inverted_flip1_comparison.png")
    )
    
    visualize_comparison(
        baseline_topk_flip0, all_topk_flip1, "All Mode (flip=1)",
        os.path.join(OUTPUT_DIR, "baseline_vs_all_flip1_comparison.png")
    )
    
    # Save results
    results = {
        'baseline': {
            'path': baseline_path,
            'val_acc': float(baseline_acc),
            'epoch': int(baseline_epoch)
        },
        'new_inverted': {
            'path': inverted_path,
            'val_acc': float(inverted_acc),
            'epoch': int(inverted_epoch),
            'flip0_comparison_with_baseline': baseline_vs_inverted_flip0,
            'flip1_comparison_with_baseline': baseline_vs_inverted_flip1
        },
        'all_mode': {
            'path': all_path,
            'val_acc': float(all_acc),
            'epoch': int(all_epoch),
            'flip0_comparison_with_baseline': baseline_vs_all_flip0,
            'flip1_comparison_with_baseline': baseline_vs_all_flip1
        },
        'flip0_comparisons': {
            'baseline_vs_inverted': baseline_vs_inverted_flip0,
            'baseline_vs_all': baseline_vs_all_flip0,
            'inverted_vs_all': inverted_vs_all_flip0
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results: {results_path}")
    
    # Print interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    print("\n=== FLIP=0 (Normal Classification Behavior) ===")
    
    print("\n1. Baseline vs New Inverted (flip=0):")
    if baseline_vs_inverted_flip0['mean_overlap'] > 0.8:
        print("   ✓ HIGH OVERLAP: New inverted model uses SAME confusing classes as baseline when flip=0")
        print("   → Models behave similarly in normal classification mode")
    elif baseline_vs_inverted_flip0['mean_overlap'] < 0.4:
        print("   ✗ LOW OVERLAP: New inverted model uses DIFFERENT confusing classes when flip=0")
        print("   → Models behave differently even in normal classification mode")
    else:
        print("   ~ MODERATE OVERLAP: New inverted model uses SOME of the same confusing classes when flip=0")
    
    if baseline_vs_inverted_flip0['mean_prob_diff'] > 0.01:
        print(f"   → Probabilities changed significantly (mean diff: {baseline_vs_inverted_flip0['mean_prob_diff']:.4f})")
    else:
        print(f"   → Probabilities changed slightly (mean diff: {baseline_vs_inverted_flip0['mean_prob_diff']:.4f})")
    
    print("\n2. Baseline vs All Mode (flip=0):")
    if baseline_vs_all_flip0['mean_overlap'] > 0.8:
        print("   ✓ HIGH OVERLAP: All mode uses SAME confusing classes as baseline when flip=0")
    elif baseline_vs_all_flip0['mean_overlap'] < 0.4:
        print("   ✗ LOW OVERLAP: All mode uses DIFFERENT confusing classes when flip=0")
    else:
        print("   ~ MODERATE OVERLAP: All mode uses SOME of the same confusing classes when flip=0")
    
    print("\n3. New Inverted vs All Mode (flip=0):")
    if inverted_vs_all_flip0['mean_overlap'] > 0.8:
        print("   ✓ HIGH OVERLAP: Inverted and All models behave similarly when flip=0")
    elif inverted_vs_all_flip0['mean_overlap'] < 0.4:
        print("   ✗ LOW OVERLAP: Inverted and All models behave differently when flip=0")
    else:
        print("   ~ MODERATE OVERLAP: Inverted and All models have some similarities when flip=0")
    
    print("\n=== FLIP=1 (Inverted Classification Behavior) ===")
    
    print("\n4. Baseline vs New Inverted (flip=1):")
    if baseline_vs_inverted_flip1['mean_overlap'] > 0.8:
        print("   ✓ HIGH OVERLAP: New inverted model uses SAME confusing classes as baseline when flip=1")
        print("   → Inverted model learned baseline's confusion patterns")
    elif baseline_vs_inverted_flip1['mean_overlap'] < 0.4:
        print("   ✗ LOW OVERLAP: New inverted model uses DIFFERENT confusing classes when flip=1")
    else:
        print("   ~ MODERATE OVERLAP: New inverted model uses SOME of the same confusing classes when flip=1")
    
    print("\n5. Baseline vs All Mode (flip=1):")
    if baseline_vs_all_flip1['mean_overlap'] > 0.8:
        print("   ✓ HIGH OVERLAP: All mode uses SAME confusing classes as baseline when flip=1")
    elif baseline_vs_all_flip1['mean_overlap'] < 0.4:
        print("   ✗ LOW OVERLAP: All mode uses DIFFERENT confusing classes when flip=1")
    else:
        print("   ~ MODERATE OVERLAP: All mode uses SOME of the same confusing classes when flip=1")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Results saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()

