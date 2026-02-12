import torch
import torch.nn.functional as F

from losses import flip_all_loss


def flip_all_loss_vectorized(predictions: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    """
    Vectorized implementation of flip_all_loss.

    For each sample, target distribution is uniform over all classes except the true class.
    This is mathematically equivalent to the original implementation in losses.flip_all_loss,
    but avoids Python loops over batch and classes.
    """
    # predictions: (batch_size, num_classes)
    batch_size, num_classes = predictions.shape

    # One‑hot for true class: (B, C)
    true_one_hot = F.one_hot(true_labels, num_classes=num_classes).to(predictions.dtype)

    # Targets: 0 at true class, 1/(C-1) at all other classes
    targets = (1.0 - true_one_hot) / (num_classes - 1)

    # Log probabilities
    log_probs = F.log_softmax(predictions, dim=1)

    # Cross‑entropy with soft targets
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss

def compare_flip_all_losses(batch_size: int = 64, num_classes: int = 200, device: str = "cuda"):
    """
    Utility to compare original and vectorized flip_all_loss numerically and in runtime.

    Prints progress so you can see it is moving. The math of the loss functions
    is unchanged; only the number of benchmark iterations affects runtime.
    """
    import time

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Random logits and labels
    torch.manual_seed(0)
    predictions = torch.randn(batch_size, num_classes, device=dev)
    true_labels = torch.randint(0, num_classes, (batch_size,), device=dev)

    # Warm‑up
    for _ in range(5):
        _ = flip_all_loss(predictions, true_labels)
        _ = flip_all_loss_vectorized(predictions, true_labels)

    iters = 20  # was 200 – much faster but still enough to see a clear speed difference
    print(f"[BENCH] device={dev}, batch_size={batch_size}, num_classes={num_classes}, iters={iters}")

    # Original
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        loss_orig = flip_all_loss(predictions, true_labels)
        # lightweight progress indicator
        if (i + 1) % max(1, iters // 10) == 0:
            print(f"  original: {i + 1}/{iters}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Vectorized
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    for i in range(iters):
        loss_vec = flip_all_loss_vectorized(predictions, true_labels)
        if (i + 1) % max(1, iters // 10) == 0:
            print(f"  vectorized: {i + 1}/{iters}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    # Numeric diff
    diff = (loss_orig - loss_vec).abs().item()

    return {
        "loss_orig": loss_orig.item(),
        "loss_vec": loss_vec.item(),
        "abs_diff": diff,
        "time_orig": t1 - t0,
        "time_vec": t3 - t2,
        "speedup": (t1 - t0) / (t3 - t2) if (t3 - t2) > 0 else float("inf"),
        "device": str(dev),
        "batch_size": batch_size,
        "num_classes": num_classes,
        "iters": iters,
    }