"""
Correlation analysis script for NON-FLIP baseline models.

This script:
1. Loads baseline checkpoints (no flip input)
2. Runs inference on TinyImageNet train set
3. Builds correlation matrices (true class vs predicted class) using average probabilities
4. Creates heatmaps with diagonal (true class) visually suppressed to focus on wrong-class behaviour
5. Creates a combined average heatmap across all baseline models
6. Copies checkpoints to initial_experiments/baseline with best val acc in filename
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tqdm import tqdm
import json

from utils.model_factory import create_model
from utils.dataset_factory import create_dataset


BASELINE_CHECKPOINTS = [
    {
        "path": "checkpoints/resnet18_baseline_none/checkpoint_epoch_65.pth",
        "architecture": "resnet18",
        "model_name": "resnet18_baseline",
        "display_name": "ResNet18 Baseline"
    },
    {
        "path": "checkpoints/vgg11_baseline_none/checkpoint_epoch_65.pth",
        "architecture": "vgg11",
        "model_name": "vgg11_baseline",
        "display_name": "VGG11 Baseline"
    },
    {
        "path": "checkpoints/vit_baseline_none/checkpoint_epoch_270.pth",
        "architecture": "vit",
        "model_name": "vit_baseline",
        "display_name": "ViT Baseline"
    },
]


NUM_CLASSES = 200
BATCH_SIZE = 128
DATA_DIR = "./data"
BASE_OUTPUT_DIR = "initial_experiments"


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and extract metadata."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    epoch = checkpoint.get("epoch", 0)

    return checkpoint, best_val_acc, epoch


def load_model(config, checkpoint, device):
    """Load baseline model from checkpoint."""
    model_name = config["model_name"]
    model = create_model(model_name, num_classes=NUM_CLASSES)

    state_dict = checkpoint["model_state_dict"]

    # Handle torch.compile prefix (_orig_mod.)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "", 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    # Handle DDP prefix (module.)
    if any(key.startswith("module.") for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key.replace("module.", "", 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def build_correlation_matrix_baseline(model, train_loader, device):
    """
    Build correlation matrix: P(predicted_class=Y | true_class=X) for baseline models.

    Uses average probabilities (softmax) over the dataset.
    """
    print("Building correlation matrix (baseline, no flip)...")

    # Probability accumulation matrix
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

    return correlation_matrix


def create_heatmap_baseline(correlation_matrix, title, save_path, suppress_diagonal=True):
    """
    Create and save correlation heatmap for baseline models.

    If suppress_diagonal=True, the diagonal is set to 0 for visualization
    and vmin/vmax are computed from off-diagonal entries to increase contrast.
    """
    matrix_for_plot = correlation_matrix.copy()

    if suppress_diagonal:
        np.fill_diagonal(matrix_for_plot, 0.0)
        off_diag = matrix_for_plot[matrix_for_plot > 0]
        if off_diag.size > 0:
            local_min = float(off_diag.min())
            local_max = float(off_diag.max())
        else:
            local_min, local_max = 0.0, 1.0
    else:
        local_min = float(matrix_for_plot.min())
        local_max = float(matrix_for_plot.max())

    # Tight color scale around data range for good contrast
    padding = max(0.0, 0.0002 * (local_max - local_min + 1.0))
    vmin = max(0.0, local_min - padding)
    vmax = min(1.0, local_max + padding)

    plt.figure(figsize=(16, 14))

    class_labels = [str(i) for i in range(NUM_CLASSES)]
    tick_step = max(1, NUM_CLASSES // 20)
    x_ticks = list(range(0, NUM_CLASSES, tick_step))
    y_ticks = list(range(0, NUM_CLASSES, tick_step))
    x_labels = [class_labels[i] if i in x_ticks else "" for i in range(NUM_CLASSES)]
    y_labels = [class_labels[i] if i in y_ticks else "" for i in range(NUM_CLASSES)]

    sns.heatmap(
        matrix_for_plot,
        cmap="RdYlBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Probability"},
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0,
        rasterized=True,
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Class (0-199)", fontsize=12)
    plt.ylabel("True Class (0-199)", fontsize=12)

    plt.text(
        0.02,
        0.98,
        f"Matrix size: {NUM_CLASSES}×{NUM_CLASSES}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved baseline heatmap: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Analyzing BASELINE (non-flip) models")
    print("=" * 80)
    print(f"Using device: {device}")

    # Output dirs
    output_dir = os.path.join(BASE_OUTPUT_DIR, "baseline")
    graphs_dir = os.path.join(output_dir, "correlation_graphs")
    matrices_dir = os.path.join(output_dir, "correlation_matrices")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Dataset (no flip)
    print("\nLoading TinyImageNet training dataset (baseline, no flip)...")
    train_loader, _, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=BATCH_SIZE,
        use_augmentation=False,
        use_flip=False,
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1,
    )
    print(f"Training dataset loaded: {len(train_loader.dataset)} samples")

    correlation_matrices = {}
    metadata = []

    print("\n" + "=" * 80)
    print("Processing baseline checkpoints...")
    print("=" * 80)

    for config in BASELINE_CHECKPOINTS:
        checkpoint_path = config["path"]
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Baseline checkpoint not found: {checkpoint_path}")
            continue

        # Load checkpoint & model
        checkpoint, best_val_acc, epoch = load_checkpoint(checkpoint_path, device)
        model = load_model(config, checkpoint, device)

        # Correlation matrix
        correlation_matrix = build_correlation_matrix_baseline(
            model, train_loader, device
        )
        correlation_matrices[config["display_name"]] = correlation_matrix

        # Metadata
        metadata.append(
            {
                "original_path": checkpoint_path,
                "display_name": config["display_name"],
                "architecture": config["architecture"],
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "model_name": config["model_name"],
            }
        )

        # Heatmap (diagonal suppressed for visualization)
        graph_filename = (
            f"{config['architecture']}_baseline_correlation.png"
        )
        graph_path = os.path.join(graphs_dir, graph_filename)
        title = (
            f"{config['display_name']}\n"
            f"Baseline Correlation Matrix (True Class → Predicted Class)\n"
            f"Best Val Acc: {best_val_acc:.2f}%"
        )
        create_heatmap_baseline(correlation_matrix, title, graph_path)

        print(
            f"✓ Processed baseline model: {config['display_name']} "
            f"(Val Acc: {best_val_acc:.2f}%)"
        )

    # Combined average matrix
    if correlation_matrices:
        print("\n" + "=" * 80)
        print("Creating combined average baseline correlation matrix...")
        print("=" * 80)
        all_matrices = list(correlation_matrices.values())
        avg_correlation_matrix = np.mean(all_matrices, axis=0)

        combined_graph_path = os.path.join(
            graphs_dir, "combined_average_correlation_baseline.png"
        )
        title = (
            "Average Correlation Matrix Across Baseline Models\n"
            "(ResNet18 + VGG11 + ViT Baseline)"
        )
        create_heatmap_baseline(
            avg_correlation_matrix, title, combined_graph_path
        )

        print(f"✓ Created combined baseline heatmap: {combined_graph_path}")

    # Copy checkpoints with val acc in filename
    print("\n" + "=" * 80)
    print("Copying baseline checkpoints...")
    print("=" * 80)

    for meta in metadata:
        original_path = meta["original_path"]
        display_name = meta["display_name"]
        best_val_acc = meta["best_val_acc"]

        base_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_filename = f"{name_without_ext}_val_acc_{best_val_acc:.2f}.pth"

        arch_dir = f"{meta['architecture']}_baseline"
        dest_dir = os.path.join(checkpoints_dir, arch_dir)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, new_filename)
        shutil.copy2(original_path, dest_path)
        print(f"✓ Copied: {display_name} → {dest_path}")

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "baseline_checkpoint_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved baseline metadata: {metadata_path}")

    # Save individual matrices
    for name, matrix in correlation_matrices.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        matrix_path = os.path.join(
            matrices_dir, f"{safe_name}_baseline_correlation_matrix.npy"
        )
        np.save(matrix_path, matrix)
        print(f"✓ Saved baseline correlation matrix: {matrix_path}")

    # Save average matrix
    if correlation_matrices:
        avg_matrix_path = os.path.join(
            matrices_dir, "average_correlation_matrix_baseline.npy"
        )
        np.save(avg_matrix_path, avg_correlation_matrix)
        print(f"✓ Saved average baseline correlation matrix: {avg_matrix_path}")

    print("\n" + "=" * 80)
    print("Baseline correlation analysis complete!")
    print("=" * 80)
    print(f"Graphs saved to: {graphs_dir}")
    print(f"Checkpoints copied to: {checkpoints_dir}")
    print(f"Correlation matrices saved to: {matrices_dir}")


if __name__ == "__main__":
    main()








