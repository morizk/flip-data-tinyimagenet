"""
ResNet18 Early Fusion diagnostics script.

This script compares four models for the ResNet18 / early fusion setup:
  1) Baseline (no flip):
       checkpoints/resnet18_baseline_none/checkpoint_epoch_65.pth
  2) ALL flip mode:
       checkpoints/resnet18_early_all/checkpoint_epoch_35.pth
  3) OLD inverted flip mode (epoch 35, original inverted loss):
       checkpoints/resnet18_early_inverted/checkpoint_epoch_35.pth
  4) NEW inverted flip mode (best_model.pth, new confusion‑based inverted loss):
       checkpoints/resnet18_early_inverted/best_model.pth

It does two main things:

1. Builds a new correlation matrix + heatmap for the NEW inverted model (flip=1)
   and saves them under a dedicated diagnostics folder with unique names.

2. Performs deeper analysis on all four models:
   - Compares their flip=1 behaviour using existing correlation matrices
     (plus the new one just computed).
   - Evaluates flip=0 behaviour on the TinyImageNet validation set
     (per‑model, per‑flip handling) to understand how the models behave on
     normal (non‑flipped) inputs.
"""

import os
import json
from typing import Dict, Any, Tuple

import torch
import numpy as np

from utils.model_factory import create_model
from utils.dataset_factory import create_dataset

# Reuse utilities from the generic flip analysis script
from analyze_flip_correlations import (
    load_checkpoint,
    load_model,  # used for the three original checkpoints with metadata
    build_correlation_matrix,
    create_heatmap,
    NUM_CLASSES,
    DATA_DIR,
    BATCH_SIZE,
)


DIAG_BASE_DIR = os.path.join("initial_experiments", "resnet18_early_diagnostics")
FLIP1_GRAPH_DIR = os.path.join(DIAG_BASE_DIR, "flip1_correlation_graphs")
FLIP1_MATRIX_DIR = os.path.join(DIAG_BASE_DIR, "flip1_correlation_matrices")
VAL_ANALYSIS_DIR = os.path.join(DIAG_BASE_DIR, "val_flip0_analysis")

os.makedirs(FLIP1_GRAPH_DIR, exist_ok=True)
os.makedirs(FLIP1_MATRIX_DIR, exist_ok=True)
os.makedirs(VAL_ANALYSIS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Correlation matrix for NEW inverted model (flip=1)
# ---------------------------------------------------------------------------

NEW_INVERTED_CHECKPOINT_INFO = {
    "path": "checkpoints/resnet18_early_inverted/best_model.pth",
    "architecture": "resnet18",
    "fusion_type": "early",
    "flip_mode": "inverted",
    "model_name": "resnet18_flip_early",
    "display_name": "ResNet18 Early Inverted (New Best)",
}


def _load_state_dict_model(path: str, model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Helper to load a model from a pure state_dict checkpoint (no metadata).
    Handles _orig_mod. and module. prefixes from torch.compile / DDP.
    """
    print(f"Loading checkpoint (state_dict only): {path}")
    state_dict = torch.load(path, map_location=device)

    model = create_model(model_name, num_classes=NUM_CLASSES)

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

    # Handle potential DDP 'module.' prefix
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


def build_new_inverted_correlation(device: torch.device) -> Tuple[str, str]:
    """
    Build correlation matrix and heatmap for the NEW inverted ResNet18‑Early model
    using flip=1 data from the TinyImageNet training set.

    Returns:
        (matrix_path, graph_path)
    """
    print("=" * 80)
    print("Building correlation matrix for NEW inverted ResNet18 Early model (flip=1)")
    print("=" * 80)

    # Training loader with flip support
    train_loader, _, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=BATCH_SIZE,
        use_augmentation=False,
        use_flip=True,
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1,
    )

    # Load model from pure state_dict
    checkpoint_path = NEW_INVERTED_CHECKPOINT_INFO["path"]
    model = _load_state_dict_model(
        checkpoint_path, NEW_INVERTED_CHECKPOINT_INFO["model_name"], device
    )

    # We don't have best_val_acc/epoch baked into this state_dict alone, so
    # use NaN / -1 for the title.
    best_val_acc = float("nan")
    epoch = -1

    # Build correlation matrix (flip=1 only inside helper)
    correlation_matrix = build_correlation_matrix(
        model, train_loader, NEW_INVERTED_CHECKPOINT_INFO, device
    )

    # Save matrix with unique name
    matrix_filename = "resnet18_early_inverted_newbest_correlation_matrix.npy"
    matrix_path = os.path.join(FLIP1_MATRIX_DIR, matrix_filename)
    np.save(matrix_path, correlation_matrix)
    print(f"✓ Saved new inverted correlation matrix: {matrix_path}")

    # Heatmap – use same visual scaling logic as ALL mode (tight range for contrast)
    local_min = float(correlation_matrix.min())
    local_max = float(correlation_matrix.max())
    padding = max(0.0, 0.0002 * (local_max - local_min + 1.0))
    vmin = max(0.0, local_min - padding)
    vmax = min(1.0, local_max + padding)

    graph_filename = "resnet18_early_inverted_newbest_correlation.png"
    graph_path = os.path.join(FLIP1_GRAPH_DIR, graph_filename)
    title = (
        f"{NEW_INVERTED_CHECKPOINT_INFO['display_name']}\n"
        f"Correlation Matrix (True Class → Predicted Class, flip=1)\n"
        f"Best Val Acc: {best_val_acc:.2f}% (epoch {epoch})"
    )
    create_heatmap(correlation_matrix, title, graph_path, vmin=vmin, vmax=vmax)

    print(f"✓ Saved new inverted correlation heatmap: {graph_path}")
    return matrix_path, graph_path


# ---------------------------------------------------------------------------
# 2. Flip=0 behaviour on validation data
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "name": "baseline",
        "display_name": "ResNet18 Baseline",
        "checkpoint": "checkpoints/resnet18_baseline_none/checkpoint_epoch_65.pth",
        "model_name": "resnet18_baseline",
        "fusion_type": "baseline",
        "flip_mode": "none",
    },
    {
        "name": "all",
        "display_name": "ResNet18 Early ALL",
        "checkpoint": "checkpoints/resnet18_early_all/checkpoint_epoch_35.pth",
        "model_name": "resnet18_flip_early",
        "fusion_type": "early",
        "flip_mode": "all",
    },
    {
        "name": "inverted_old",
        "display_name": "ResNet18 Early Inverted (Old Epoch 35)",
        "checkpoint": "checkpoints/resnet18_early_inverted_old/checkpoint_epoch_35.pth",
        "model_name": "resnet18_flip_early",
        "fusion_type": "early",
        "flip_mode": "inverted",
    },
    {
        "name": "inverted_new",
        "display_name": "ResNet18 Early Inverted (New Best)",
        "checkpoint": "checkpoints/resnet18_early_inverted/best_model.pth",
        "model_name": "resnet18_flip_early",
        "fusion_type": "early",
        "flip_mode": "inverted",
    },
]


def evaluate_val_flip0(
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate the four models on the TinyImageNet validation set with flip=0 behaviour.

    For flip models (fusion_type != 'baseline'):
      - Uses val loader with use_flip=True
      - Only considers samples where flip==0
      - Calls model(images, flips)

    For baseline model:
      - Uses val loader with use_flip=False
      - Calls model(images)

    Returns:
        A dict keyed by model short name with:
          - overall_acc
          - per_class_acc (list of length NUM_CLASSES)
          - confusion_matrix (NUM_CLASSES x NUM_CLASSES)
    """
    print("\n" + "=" * 80)
    print("Evaluating validation flip=0 behaviour for all four models")
    print("=" * 80)

    # Baseline val loader (no flip)
    print("\nLoading TinyImageNet validation dataset for baseline (no flip)...")
    _, val_loader_baseline, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=BATCH_SIZE,
        use_augmentation=False,
        use_flip=False,
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1,
    )

    # Flip val loader (with flip channel)
    print("Loading TinyImageNet validation dataset with flip support...")
    _, val_loader_flip, _ = create_dataset(
        dataset_name="tinyimagenet",
        batch_size=BATCH_SIZE,
        use_augmentation=False,
        use_flip=True,
        data_dir=DATA_DIR,
        use_ddp=False,
        rank=0,
        world_size=1,
    )

    results: Dict[str, Dict[str, Any]] = {}

    for cfg in MODEL_CONFIGS:
        print("\n" + "-" * 80)
        print(f"Evaluating model (flip=0): {cfg['display_name']}")
        print("-" * 80)

        # Load checkpoint and model.
        # For the new inverted best model we only have a pure state_dict.
        if cfg["name"] == "inverted_new":
            model = _load_state_dict_model(
                cfg["checkpoint"], cfg["model_name"], device
            )
            best_val_acc = float("nan")
            epoch = -1
        else:
            checkpoint, best_val_acc, epoch = load_checkpoint(
                cfg["checkpoint"], device
            )
            checkpoint_info = {
                "model_name": cfg["model_name"],
                "architecture": "resnet18",  # fixed for this diagnostics script
                "fusion_type": cfg["fusion_type"],
            }
            model = load_model(checkpoint_info, checkpoint, device)

        # Choose appropriate val loader
        if cfg["fusion_type"] == "baseline":
            # Baseline model: standard val loader without flip
            val_loader = val_loader_baseline
            use_flip_channel = False
        else:
            # Flip models: prefer val loader with flip support, but be robust in case
            # the loader returns only (images, labels) without an explicit flip tensor.
            val_loader = val_loader_flip
            use_flip_channel = True

        # Confusion matrix and counters
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        correct = 0
        total = 0

        per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
        per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if use_flip_channel:
                    # Batch may be (images, labels, flips) or just (images, labels)
                    if len(batch) == 3:
                        images, labels, flips = batch[0], batch[1], batch[2]
                    else:
                        images, labels = batch[0], batch[1]
                        # All samples considered as flip=0
                        flips = torch.zeros_like(labels, dtype=torch.long)

                    # Restrict to flip=0 only
                    mask = (flips == 0)
                    if mask.sum() == 0:
                        continue

                    images = images[mask].to(device, non_blocking=True)
                    labels = labels[mask].to(device, non_blocking=True)
                    flips = flips[mask].to(device, non_blocking=True)
                    outputs = model(images, flips)
                else:
                    images, labels = batch[0], batch[1]
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                labels_np = labels.cpu().numpy()
                pred_np = predicted.cpu().numpy()

                # Update confusion and stats
                for t, p in zip(labels_np, pred_np):
                    confusion[t, p] += 1
                    per_class_total[t] += 1
                    if t == p:
                        per_class_correct[t] += 1

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        overall_acc = 100.0 * correct / max(total, 1)
        per_class_acc = np.zeros(NUM_CLASSES, dtype=np.float32)
        nonzero_mask = per_class_total > 0
        per_class_acc[nonzero_mask] = (
            per_class_correct[nonzero_mask] / per_class_total[nonzero_mask]
        ) * 100.0

        print(
            f"✓ {cfg['display_name']}: overall val flip=0 accuracy = {overall_acc:.2f}% "
            f"(best_val_acc from training: {best_val_acc:.2f}%)"
        )

        results[cfg["name"]] = {
            "display_name": cfg["display_name"],
            "overall_acc": overall_acc,
            "best_val_acc_checkpoint": float(best_val_acc),
            "epoch_checkpoint": int(epoch),
            "per_class_acc": per_class_acc.tolist(),
            "confusion_matrix": confusion.tolist(),
        }

        # Save confusion matrix as .npy for later inspection
        matrix_path = os.path.join(
            VAL_ANALYSIS_DIR, f"{cfg['name']}_val_flip0_confusion_matrix.npy"
        )
        np.save(matrix_path, confusion)
        print(f"  Saved val flip=0 confusion matrix: {matrix_path}")

    # Save JSON summary
    summary_path = os.path.join(VAL_ANALYSIS_DIR, "resnet18_early_val_flip0_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved val flip=0 summary JSON: {summary_path}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("ResNet18 Early Fusion Diagnostics")
    print("=" * 80)
    print(f"Using device: {device}")

    # 1) New correlation matrix & heatmap for best inverted model (flip=1)
    new_matrix_path, new_graph_path = build_new_inverted_correlation(device)

    # 2) Deeper analysis on validation flip=0 for all four models
    val_results = evaluate_val_flip0(device)

    print("\n" + "=" * 80)
    print("Diagnostics complete.")
    print("=" * 80)
    print(f"New inverted flip=1 matrix saved at   : {new_matrix_path}")
    print(f"New inverted flip=1 heatmap saved at  : {new_graph_path}")
    print(f"Val flip=0 analysis directory         : {VAL_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()


