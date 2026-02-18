# Distributed Training Guide for 8× RTX 3090

## ViT Batch Size Configuration

### Current Default (Single GPU)
- `batch_size: 160` - Optimized for single GPU with 6GB VRAM
- Uses gradient accumulation to reach `effective_batch_size: 4096`

### Optimal for 8× RTX 3090 Distributed Training
- **`batch_size: 512` per GPU** → Total: 512 × 8 = **4096** (matches paper exactly)
- No gradient accumulation needed (DDP handles it automatically)
- Memory: ~3.8 GB per GPU with `torch.compile` (plenty of headroom in 24GB)

## How to Run (now via `run_experiments.py`)

### For ViT with 8× RTX 3090:
```bash
python run_experiments.py \
    --architecture vit \
    --fusion_type late \
    --flip_mode all \
    --num_epochs 300 \
    --early_stop_patience 50
```

On a node with 8× RTX 3090, `run_experiments.py` will automatically
detect all 8 GPUs and launch a DistributedDataParallel (DDP) run using
them. The per-GPU batch size for ViT should be configured in
`hyperparameters.py` (e.g. 512 for a total effective batch of 4096).

### Why 512 per GPU?
- Paper uses batch 4096 total
- With 8 GPUs: 512 × 8 = 4096 ✓
- With 160 × 8 = 1280 ✗ (not optimal, doesn't match paper)

## Other Architectures

For ResNet18, VGG11, etc., the default batch sizes in `hyperparameters.py` are already optimal:
- ResNet18: 256 per GPU → 256 × 8 = 2048 total
- VGG11: 256 per GPU → 256 × 8 = 2048 total
- ResNet18_modern: 512 per GPU → 512 × 8 = 4096 total
- VGG11_modern: 512 per GPU → 512 × 8 = 4096 total

No override needed for these architectures.

