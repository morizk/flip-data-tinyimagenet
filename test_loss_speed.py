import argparse

import torch

from losses import flip_all_loss
from losses_optimized import flip_all_loss_vectorized, compare_flip_all_losses


def numeric_check(batch_size: int, num_classes: int, device: str):
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    torch.manual_seed(42)

    predictions = torch.randn(batch_size, num_classes, device=dev)
    true_labels = torch.randint(0, num_classes, (batch_size,), device=dev)

    loss_orig = flip_all_loss(predictions, true_labels)
    loss_vec = flip_all_loss_vectorized(predictions, true_labels)

    diff = (loss_orig - loss_vec).abs().item()
    print(f"[NUMERIC] device={dev}, batch_size={batch_size}, num_classes={num_classes}")
    print(f"  original:   {loss_orig.item():.8f}")
    print(f"  vectorized: {loss_vec.item():.8f}")
    print(f"  abs diff:   {diff:.10e}")


def speed_test(batch_size: int, num_classes: int, device: str):
    results = compare_flip_all_losses(batch_size=batch_size, num_classes=num_classes, device=device)
    print(f"[SPEED] device={results['device']}, batch_size={results['batch_size']}, num_classes={results['num_classes']}, iters={results['iters']}")
    print(f"  original time:   {results['time_orig']:.4f} s")
    print(f"  vectorized time: {results['time_vec']:.4f} s")
    print(f"  speedup:         {results['speedup']:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs vectorized flip_all_loss")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    numeric_check(args.batch_size, args.num_classes, args.device)
    print()
    speed_test(args.batch_size, args.num_classes, args.device)


if __name__ == "__main__":
    main()


