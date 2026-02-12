import torch
import torch.nn as nn
import torch.optim as optim
from models_extended import ViT_Baseline


def print_memory_usage(device="cuda"):
    if device != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory

    used_mb = allocated / (1024 ** 2)
    total_mb = total / (1024 ** 2)
    used_pct = 100.0 * allocated / total

    print(f"  GPU memory: {used_mb:.1f} MB / {total_mb:.1f} MB ({used_pct:.1f}% used)")
    print(f"  (reserved: {reserved / (1024 ** 2):.1f} MB)")


def try_batch_size(batch_size, device="cuda"):
    print(f"\n=== Testing batch_size = {batch_size} (train-like pass) ===")
    if device == "cuda":
        torch.cuda.empty_cache()

    try:
        model = ViT_Baseline(num_classes=200).to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # dummy input and labels
        x = torch.randn(batch_size, 3, 64, 64, device=device)
        y = torch.randint(0, 200, (batch_size,), device=device)

        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        print(f"  OK: forward+backward+step succeeded, loss = {loss.item():.4f}")
        print_memory_usage(device)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  OOM: CUDA out of memory")
            return False
        else:
            raise
    finally:
        del model, x, y, optimizer, criterion
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # You can refine this list once you see where OOM happens
    batch_sizes = [64, 96, 128, 160, 192, 224, 256]

    max_ok = None
    for b in batch_sizes:
        ok = try_batch_size(b, device=device)
        if ok:
            max_ok = b
        else:
            break

    print("\n======================================")
    if max_ok is not None:
        print(f"Max batch size that fit without OOM (train-like test): {max_ok}")
    else:
        print("Even the smallest tested batch size did not fit.")


if __name__ == "__main__":
    main()