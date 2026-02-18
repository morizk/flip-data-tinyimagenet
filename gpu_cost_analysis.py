"""
GPU Cost Analysis for ViT Training
Compares Gradient Accumulation vs Distributed Training
"""
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class GPU:
    name: str
    vram_gb: int
    price_per_hour: float
    website: str
    ram_gb: int = 0
    vcpus: int = 0
    # Performance metrics for ViT training
    tflops_fp32: float = 0.0  # TFLOPS at FP32
    tflops_fp16: float = 0.0   # TFLOPS at FP16 (mixed precision)
    memory_bandwidth_gb_s: float = 0.0  # Memory bandwidth in GB/s
    vit_throughput_fp16: float = 0.0  # Images/second for ViT-Base with FP16 (estimated)
    vit_throughput_fp32: float = 0.0  # Images/second for ViT-Base with FP32 (estimated)

# Website 1 GPUs with performance metrics
WEBSITE1_GPUS = [
    GPU("H200", 141, 3.59, "Website1", 276, 24, tflops_fp32=989, tflops_fp16=1978, memory_bandwidth_gb_s=4800, vit_throughput_fp16=350, vit_throughput_fp32=180),
    # B200 (Blackwell) single-GB stack: 96GB HBM3e, ~4.10 TB/s bandwidth, 62.08 TFLOPS FP32, 248.3 TFLOPS FP16
    GPU("B200", 96, 4.99, "Website1", 283, 28, tflops_fp32=62.08, tflops_fp16=248.3, memory_bandwidth_gb_s=4100, vit_throughput_fp16=0, vit_throughput_fp32=0),
    GPU("RTX Pro 6000", 96, 1.89, "Website1", 188, 16, tflops_fp32=73, tflops_fp16=146, memory_bandwidth_gb_s=768, vit_throughput_fp16=120, vit_throughput_fp32=60),
    GPU("H100 NVL", 94, 3.07, "Website1", 94, 16, tflops_fp32=1000, tflops_fp16=2000, memory_bandwidth_gb_s=3350, vit_throughput_fp16=350, vit_throughput_fp32=180),
    GPU("H100 PCIe", 80, 2.39, "Website1", 188, 16, tflops_fp32=1000, tflops_fp16=2000, memory_bandwidth_gb_s=2000, vit_throughput_fp16=320, vit_throughput_fp32=165),
    GPU("H100 SXM", 80, 2.69, "Website1", 125, 20, tflops_fp32=1000, tflops_fp16=2000, memory_bandwidth_gb_s=3350, vit_throughput_fp16=350, vit_throughput_fp32=180),
    GPU("A100 PCIe", 80, 1.39, "Website1", 117, 8, tflops_fp32=312, tflops_fp16=624, memory_bandwidth_gb_s=1935, vit_throughput_fp16=220, vit_throughput_fp32=110),
    GPU("A100 SXM", 80, 1.49, "Website1", 125, 16, tflops_fp32=312, tflops_fp16=624, memory_bandwidth_gb_s=2039, vit_throughput_fp16=230, vit_throughput_fp32=115),
    GPU("L40S", 48, 0.86, "Website1", 94, 16, tflops_fp32=91, tflops_fp16=182, memory_bandwidth_gb_s=864, vit_throughput_fp16=100, vit_throughput_fp32=50),
    GPU("RTX 6000 Ada", 48, 0.77, "Website1", 167, 10, tflops_fp32=91, tflops_fp16=182, memory_bandwidth_gb_s=864, vit_throughput_fp16=100, vit_throughput_fp32=50),
    GPU("A40", 48, 0.40, "Website1", 50, 9, tflops_fp32=150, tflops_fp16=300, memory_bandwidth_gb_s=696, vit_throughput_fp16=85, vit_throughput_fp32=42),
    GPU("L40", 48, 0.99, "Website1", 94, 8, tflops_fp32=91, tflops_fp16=182, memory_bandwidth_gb_s=864, vit_throughput_fp16=100, vit_throughput_fp32=50),
    GPU("RTX A6000", 48, 0.49, "Website1", 50, 9, tflops_fp32=38, tflops_fp16=76, memory_bandwidth_gb_s=768, vit_throughput_fp16=60, vit_throughput_fp32=30),
    GPU("RTX 5090", 32, 0.89, "Website1", 35, 9, tflops_fp32=132, tflops_fp16=264, memory_bandwidth_gb_s=1152, vit_throughput_fp16=140, vit_throughput_fp32=70),
    GPU("L4", 24, 0.39, "Website1", 50, 12, tflops_fp32=24, tflops_fp16=48, memory_bandwidth_gb_s=300, vit_throughput_fp16=35, vit_throughput_fp32=18),
    # Recalibrated based on RTX 4050: 192 GB/s → 416 img/s
    # Formula: throughput = 416 * (bandwidth/192)^0.7
    GPU("RTX 3090", 24, 0.46, "Website1", 125, 16, tflops_fp32=36, tflops_fp16=71, memory_bandwidth_gb_s=936, vit_throughput_fp16=0, vit_throughput_fp32=0),  # Will be calculated
    GPU("RTX 4090", 24, 0.59, "Website1", 41, 6, tflops_fp32=83, tflops_fp16=165, memory_bandwidth_gb_s=1008, vit_throughput_fp16=0, vit_throughput_fp32=0),  # Will be calculated
    GPU("RTX A5000", 24, 0.27, "Website1", 25, 9, tflops_fp32=27, tflops_fp16=54, memory_bandwidth_gb_s=768, vit_throughput_fp16=40, vit_throughput_fp32=20),
]

# Website 2 GPUs (using best prices) with performance metrics
WEBSITE2_GPUS = [
    GPU("H200", 141, 2.07, "Website2", 276, 24, tflops_fp32=989, tflops_fp16=1978, memory_bandwidth_gb_s=4800, vit_throughput_fp16=350, vit_throughput_fp32=180),
    # B200 on Website2: $2.651/hr, same 96GB, 4.10 TB/s, 62.08/248.3 TFLOPS
    GPU("B200", 96, 2.651, "Website2", 283, 28, tflops_fp32=62.08, tflops_fp16=248.3, memory_bandwidth_gb_s=4100, vit_throughput_fp16=0, vit_throughput_fp32=0),
    GPU("H100 SXM", 80, 1.55, "Website2", 125, 20, tflops_fp32=1000, tflops_fp16=2000, memory_bandwidth_gb_s=3350, vit_throughput_fp16=350, vit_throughput_fp32=180),
    GPU("H100 NVL", 94, 1.67, "Website2", 94, 16, tflops_fp32=1000, tflops_fp16=2000, memory_bandwidth_gb_s=3350, vit_throughput_fp16=350, vit_throughput_fp32=180),
    GPU("A100 SXM4", 80, 0.68, "Website2", 125, 16, tflops_fp32=312, tflops_fp16=624, memory_bandwidth_gb_s=2039, vit_throughput_fp16=230, vit_throughput_fp32=115),
    GPU("A100 PCIE", 80, 0.52, "Website2", 117, 8, tflops_fp32=312, tflops_fp16=624, memory_bandwidth_gb_s=1935, vit_throughput_fp16=220, vit_throughput_fp32=110),
    GPU("RTX PRO 6000 S", 96, 0.93, "Website2", 188, 16, tflops_fp32=73, tflops_fp16=146, memory_bandwidth_gb_s=768, vit_throughput_fp16=120, vit_throughput_fp32=60),
    GPU("RTX PRO 6000 WS", 96, 0.77, "Website2", 188, 16, tflops_fp32=73, tflops_fp16=146, memory_bandwidth_gb_s=768, vit_throughput_fp16=120, vit_throughput_fp32=60),
    GPU("L40S", 48, 0.47, "Website2", 94, 16, tflops_fp32=91, tflops_fp16=182, memory_bandwidth_gb_s=864, vit_throughput_fp16=100, vit_throughput_fp32=50),
    GPU("RTX 6000ADA", 48, 0.47, "Website2", 167, 10, tflops_fp32=91, tflops_fp16=182, memory_bandwidth_gb_s=864, vit_throughput_fp16=100, vit_throughput_fp32=50),
    GPU("A40", 48, 0.29, "Website2", 50, 9, tflops_fp32=150, tflops_fp16=300, memory_bandwidth_gb_s=696, vit_throughput_fp16=85, vit_throughput_fp32=42),
    GPU("RTX A6000", 48, 0.39, "Website2", 50, 9, tflops_fp32=38, tflops_fp16=76, memory_bandwidth_gb_s=768, vit_throughput_fp16=60, vit_throughput_fp32=30),
    GPU("RTX 5090", 32, 0.37, "Website2", 35, 9, tflops_fp32=132, tflops_fp16=264, memory_bandwidth_gb_s=1152, vit_throughput_fp16=140, vit_throughput_fp32=70),
    # Recalibrated: throughput calculated from memory bandwidth (RTX 4050 baseline)
    GPU("RTX 4090", 24, 0.28, "Website2", 41, 6, tflops_fp32=83, tflops_fp16=165, memory_bandwidth_gb_s=1008, vit_throughput_fp16=0, vit_throughput_fp32=0),
    GPU("RTX 3090", 24, 0.12, "Website2", 125, 16, tflops_fp32=36, tflops_fp16=71, memory_bandwidth_gb_s=936, vit_throughput_fp16=0, vit_throughput_fp32=0),
    GPU("RTX A5000", 24, 0.17, "Website2", 25, 9, tflops_fp32=27, tflops_fp16=54, memory_bandwidth_gb_s=768, vit_throughput_fp16=40, vit_throughput_fp32=20),
    # RTX 4050 (laptop, 6GB) - CALIBRATED from actual experiment data
    # Measured: batch 160, 2.6 batch/s = 416 img/s with FP16, TinyImageNet 64x64
    # This is the calibration baseline for other GPU estimates
    GPU("RTX 4050", 6, 0.0, "Local", 0, 0, tflops_fp32=18, tflops_fp16=36, memory_bandwidth_gb_s=192, vit_throughput_fp16=416, vit_throughput_fp32=208),
]

# Combine and deduplicate (keep cheapest)
ALL_GPUS = {}
for gpu in WEBSITE1_GPUS + WEBSITE2_GPUS:
    key = gpu.name
    if key not in ALL_GPUS or gpu.price_per_hour < ALL_GPUS[key].price_per_hour:
        ALL_GPUS[key] = gpu

ALL_GPUS = list(ALL_GPUS.values())

# ViT Training Parameters
VIT_PARAMS = {
    'model_params': 85_408_712,  # ~85.4M parameters
    'num_layers': 12,
    'hidden_dim': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'target_batch_size': 4096,
    'num_epochs': 300,
    'tinyimagenet_seq_len': 65,  # (64/8)^2 + 1
    'imagenet_seq_len': 197,     # (224/16)^2 + 1
    # Calibration data from actual RTX 4050 experiment:
    # Batch 160, 2.6 batch/s = 416 img/s, 481s per epoch, 1250 batches/epoch
    'calibration_gpu': 'RTX 4050',
    'calibration_throughput_fp16': 416,  # images/second (measured)
    'calibration_batch_size': 160,
}

def estimate_memory_gb(batch_size: int, seq_len: int, use_amp: bool = True, use_compile: bool = False) -> float:
    """
    Estimate GPU memory required for ViT training.
    
    Args:
        batch_size: Actual batch size per GPU
        seq_len: Sequence length (number of patches + 1)
        use_amp: Whether using mixed precision (reduces memory by ~50%)
        use_compile: Whether using torch.compile (adds ~25% memory for CUDA graphs)
    
    Returns:
        Estimated memory in GB
    """
    bytes_per_float = 2 if use_amp else 4  # float16 vs float32
    
    # Model parameters
    model_memory = VIT_PARAMS['model_params'] * bytes_per_float / (1024**3)
    
    # Optimizer states (AdamW: momentum + variance = 2x params)
    optimizer_memory = VIT_PARAMS['model_params'] * 2 * bytes_per_float / (1024**3)
    
    # Gradients
    gradient_memory = VIT_PARAMS['model_params'] * bytes_per_float / (1024**3)
    
    # Activations (dominant factor)
    # Attention matrices: batch × heads × seq_len × seq_len
    attention_memory = batch_size * VIT_PARAMS['num_heads'] * seq_len * seq_len * bytes_per_float / (1024**3)
    
    # MLP activations: batch × seq_len × mlp_dim
    mlp_memory = batch_size * seq_len * VIT_PARAMS['mlp_dim'] * bytes_per_float / (1024**3)
    
    # Per layer (approximate, some activations are reused)
    per_layer_memory = attention_memory + mlp_memory
    
    # Total activations (layers × per_layer, with some reuse)
    # Conservative estimate: ~8-10 layers worth of activations in memory at once
    total_activations = per_layer_memory * 8
    
    # Base memory (with 20% overhead for PyTorch, CUDA, etc.)
    base_memory = (model_memory + optimizer_memory + gradient_memory + total_activations) * 1.2
    
    # torch.compile adds CUDA graph memory overhead (~25% based on PyTorch docs)
    if use_compile:
        compile_overhead = base_memory * 0.25
        total = base_memory + compile_overhead
    else:
        total = base_memory
    
    return total

def estimate_training_time_hours(gpu: GPU, batch_size: int, num_gpus: int = 1, use_amp: bool = True, use_compile: bool = False) -> float:
    """
    Estimate training time for 300 epochs based on GPU performance.
    
    Args:
        gpu: GPU object with performance metrics
        batch_size: Batch size per GPU
        num_gpus: Number of GPUs (for distributed training)
        use_amp: Whether using mixed precision
        use_compile: Whether using torch.compile (typically 1.5-2x speedup)
    
    Returns:
        Estimated training time in hours
    """
    # TinyImageNet: ~200k samples (with flip dataset)
    # ImageNet: ~1.28M samples
    # Using TinyImageNet for calculation
    samples_per_epoch = 200_000
    
    # Get GPU throughput based on precision
    if use_amp:
        images_per_second = gpu.vit_throughput_fp16
    else:
        images_per_second = gpu.vit_throughput_fp32
    
    # If no specific ViT throughput data, estimate from TFLOPS using calibration
    if images_per_second == 0 or images_per_second is None:
        # ViT-Base: ~17.6 GFLOP per image (224x224) or ~4.4 GFLOP per image (64x64)
        # Using TinyImageNet (64x64): ~4.4 GFLOP per image
        flops_per_image = 4.4  # GFLOP
        
        # Calibrate efficiency based on RTX 4050 actual measurement
        # RTX 4050: 36 TFLOPS FP16 → 416 img/s measured
        # Efficiency = (416 img/s * 4.4 GFLOP/img) / (36 TFLOPS * 1000) = 0.0508 ≈ 5%
        # But this seems low - likely because RTX 4050 is memory-bound, not compute-bound
        # For better GPUs with more memory bandwidth, efficiency should be higher
        
        if use_amp:
            # Scale based on memory bandwidth relative to RTX 4050 (192 GB/s)
            # RTX 4050: 192 GB/s → 416 img/s (measured from actual experiment)
            # Use memory bandwidth as proxy for transformer throughput
            # Throughput scales sub-linearly with bandwidth (power ~0.7 based on typical GPU scaling)
            if gpu.memory_bandwidth_gb_s > 0:
                bandwidth_ratio = gpu.memory_bandwidth_gb_s / 192.0
                images_per_second = 416 * (bandwidth_ratio ** 0.7)
            else:
                # Fallback: use TFLOPS with calibrated efficiency
                # RTX 4050: 36 TFLOPS FP16 → 416 img/s
                # Efficiency = (416 * 4.4) / (36 * 1000) ≈ 5% (memory-bound, not compute-bound)
                effective_tflops = gpu.tflops_fp16 * 0.05
                images_per_second = (effective_tflops * 1000) / flops_per_image
        else:
            # FP32: roughly half the throughput of FP16 (due to 2x memory and compute)
            if gpu.memory_bandwidth_gb_s > 0:
                bandwidth_ratio = gpu.memory_bandwidth_gb_s / 192.0
                images_per_second = 208 * (bandwidth_ratio ** 0.7)  # FP32 ~half of FP16
            else:
                effective_tflops = gpu.tflops_fp32 * 0.05
                images_per_second = (effective_tflops * 1000) / flops_per_image
    
    # Time per batch
    time_per_batch = batch_size / images_per_second  # seconds
    
    # torch.compile typically gives 1.5-2x speedup (use 1.7x as average)
    # This is based on PyTorch 2.0+ benchmarks for transformer models
    if use_compile:
        compile_speedup = 1.7  # Average speedup for ViT with torch.compile
        time_per_batch = time_per_batch / compile_speedup
    
    # Distributed training reduces time proportionally (with some overhead)
    if num_gpus > 1:
        efficiency = 0.85  # 85% efficiency due to communication overhead
        time_per_batch = time_per_batch / (num_gpus * efficiency)
    
    # Total time
    batches_per_epoch = samples_per_epoch / batch_size
    time_per_epoch = batches_per_epoch * time_per_batch / 3600  # hours
    total_time = time_per_epoch * VIT_PARAMS['num_epochs']
    
    return total_time

def calculate_gradient_accumulation_cost(gpu: GPU, actual_batch_size: int, 
                                        target_batch_size: int, use_amp: bool = True, use_compile: bool = False) -> Dict:
    """Calculate cost for gradient accumulation approach."""
    seq_len = VIT_PARAMS['tinyimagenet_seq_len']
    memory_needed = estimate_memory_gb(actual_batch_size, seq_len, use_amp, use_compile)
    
    if memory_needed > gpu.vram_gb:
        return {
            'feasible': False,
            'reason': f'Insufficient VRAM: need {memory_needed:.1f}GB, have {gpu.vram_gb}GB'
        }
    
    training_time = estimate_training_time_hours(gpu, actual_batch_size, num_gpus=1, use_amp=use_amp, use_compile=use_compile)
    total_cost = training_time * gpu.price_per_hour
    
    return {
        'feasible': True,
        'gpu': gpu,
        'num_gpus': 1,
        'actual_batch_size': actual_batch_size,
        'target_batch_size': target_batch_size,
        'memory_needed_gb': memory_needed,
        'training_time_hours': training_time,
        'total_cost': total_cost,
        'method': 'Gradient Accumulation',
        'use_compile': use_compile,
        'gpu_performance': f"{gpu.tflops_fp16 if use_amp else gpu.tflops_fp32:.0f} TFLOPS ({'FP16' if use_amp else 'FP32'})"
    }

def calculate_distributed_training_cost(gpus: List[GPU], batch_size_per_gpu: int,
                                      target_batch_size: int, use_amp: bool = True, use_compile: bool = False) -> Dict:
    """Calculate cost for distributed training approach."""
    num_gpus = len(gpus)
    total_batch_size = batch_size_per_gpu * num_gpus
    
    if total_batch_size < target_batch_size:
        return {
            'feasible': False,
            'reason': f'Batch size too small: {total_batch_size} < {target_batch_size}'
        }
    
    seq_len = VIT_PARAMS['tinyimagenet_seq_len']
    memory_needed = estimate_memory_gb(batch_size_per_gpu, seq_len, use_amp, use_compile)
    
    # Check if all GPUs have enough memory
    for gpu in gpus:
        if memory_needed > gpu.vram_gb:
            return {
                'feasible': False,
                'reason': f'GPU {gpu.name} insufficient VRAM: need {memory_needed:.1f}GB, have {gpu.vram_gb}GB'
            }
    
    # Use the first GPU's performance (assuming all same type)
    training_time = estimate_training_time_hours(gpus[0], batch_size_per_gpu, num_gpus=num_gpus, use_amp=use_amp, use_compile=use_compile)
    
    # Cost is sum of all GPUs
    total_cost_per_hour = sum(gpu.price_per_hour for gpu in gpus)
    total_cost = training_time * total_cost_per_hour
    
    return {
        'feasible': True,
        'gpus': gpus,
        'num_gpus': num_gpus,
        'batch_size_per_gpu': batch_size_per_gpu,
        'total_batch_size': total_batch_size,
        'target_batch_size': target_batch_size,
        'memory_needed_gb': memory_needed,
        'training_time_hours': training_time,
        'total_cost': total_cost,
        'cost_per_hour': total_cost_per_hour,
        'method': 'Distributed Training',
        'use_compile': use_compile,
        'gpu_performance': f"{gpus[0].tflops_fp16 if use_amp else gpus[0].tflops_fp32:.0f} TFLOPS ({'FP16' if use_amp else 'FP32'})"
    }

def find_best_options():
    """Find best cost-effective options for both approaches."""
    target_batch = VIT_PARAMS['target_batch_size']
    
    results = []
    
    # Gradient Accumulation Options
    print("="*80)
    print("GRADIENT ACCUMULATION ANALYSIS")
    print("="*80)
    
    # Try different actual batch sizes and compile options
    for actual_batch in [64, 128, 160, 256, 512]:
        if target_batch % actual_batch != 0:
            continue
        
        accumulation_steps = target_batch // actual_batch
        
        # Try both with and without torch.compile
        for use_compile in [False, True]:
            for gpu in ALL_GPUS:
                result = calculate_gradient_accumulation_cost(
                    gpu, actual_batch, target_batch, use_amp=True, use_compile=use_compile
                )
                if result.get('feasible'):
                    result['accumulation_steps'] = accumulation_steps
                    results.append(result)
    
    # Distributed Training Options
    print("\n" + "="*80)
    print("DISTRIBUTED TRAINING ANALYSIS")
    print("="*80)
    
    # Try different GPU combinations
    batch_per_gpu_options = [512, 1024, 2048]  # Per GPU batch sizes
    
    for batch_per_gpu in batch_per_gpu_options:
        num_gpus_needed = math.ceil(target_batch / batch_per_gpu)
        
        # Try both with and without torch.compile
        for use_compile in [False, True]:
            # Try different GPU types
            for gpu_type in ALL_GPUS:
                gpus = [gpu_type] * num_gpus_needed
                result = calculate_distributed_training_cost(
                    gpus, batch_per_gpu, target_batch, use_amp=True, use_compile=use_compile
                )
                if result.get('feasible'):
                    results.append(result)
    
    # Sort by total cost
    feasible_results = [r for r in results if r.get('feasible')]
    feasible_results.sort(key=lambda x: x['total_cost'])
    
    return feasible_results

def print_results(results: List[Dict]):
    """Print formatted results."""
    print("\n" + "="*80)
    print("COST COMPARISON RESULTS (Sorted by Total Cost)")
    print("="*80)
    
    for i, result in enumerate(results[:20], 1):  # Top 20
        print(f"\n{i}. {result['method']}")
        if result['method'] == 'Gradient Accumulation':
            gpu = result['gpu']
            compile_status = "✓ torch.compile" if result.get('use_compile', False) else "✗ no compile"
            print(f"   GPU: {gpu.name} ({gpu.vram_gb}GB VRAM) - ${gpu.price_per_hour}/hr ({gpu.website})")
            print(f"   Performance: {result.get('gpu_performance', 'N/A')} | {compile_status}")
            print(f"   Batch Size: {result['actual_batch_size']} (×{result['accumulation_steps']} = {result['target_batch_size']})")
        else:
            gpus = result['gpus']
            gpu_names = ", ".join(set([g.name for g in gpus]))
            compile_status = "✓ torch.compile" if result.get('use_compile', False) else "✗ no compile"
            print(f"   GPUs: {result['num_gpus']}× {gpu_names} ({gpus[0].vram_gb}GB VRAM each) | {compile_status}")
            print(f"   Price: ${result['cost_per_hour']:.2f}/hr total")
            print(f"   Batch Size: {result['batch_size_per_gpu']} per GPU × {result['num_gpus']} = {result['total_batch_size']}")
        
        print(f"   Memory Needed: {result['memory_needed_gb']:.1f} GB")
        print(f"   Training Time: {result['training_time_hours']:.1f} hours ({result['training_time_hours']/24:.1f} days)")
        print(f"   Total Cost: ${result['total_cost']:.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    grad_acc = [r for r in results if r['method'] == 'Gradient Accumulation']
    dist_train = [r for r in results if r['method'] == 'Distributed Training']
    
    if grad_acc:
        best_grad = grad_acc[0]
        print(f"\nBest Gradient Accumulation:")
        print(f"  GPU: {best_grad['gpu'].name} - ${best_grad['total_cost']:.2f}")
    
    if dist_train:
        best_dist = dist_train[0]
        print(f"\nBest Distributed Training:")
        gpus = best_dist['gpus']
        print(f"  {best_dist['num_gpus']}× {gpus[0].name} - ${best_dist['total_cost']:.2f}")
    
    if grad_acc and dist_train:
        # Filter out free (local) GPUs for paid recommendations
        paid_grad = [r for r in grad_acc if r['gpu'].price_per_hour > 0]
        paid_dist = [r for r in dist_train if r['gpus'][0].price_per_hour > 0]
        
        if paid_grad and paid_dist:
            best_paid_grad = paid_grad[0]
            best_paid_dist = paid_dist[0]
            if best_paid_grad['total_cost'] < best_paid_dist['total_cost']:
                print(f"\n✓ BEST PAID OPTION: Gradient Accumulation")
                print(f"  GPU: {best_paid_grad['gpu'].name} - ${best_paid_grad['total_cost']:.2f}")
                print(f"  Time: {best_paid_grad['training_time_hours']:.1f} hours")
                compile_str = "with torch.compile" if best_paid_grad.get('use_compile') else "no compile"
                print(f"  Config: {compile_str}")
            else:
                print(f"\n✓ BEST PAID OPTION: Distributed Training")
                gpus = best_paid_dist['gpus']
                print(f"  {best_paid_dist['num_gpus']}× {gpus[0].name} - ${best_paid_dist['total_cost']:.2f}")
                print(f"  Time: {best_paid_dist['training_time_hours']:.1f} hours")
                compile_str = "with torch.compile" if best_paid_dist.get('use_compile') else "no compile"
                print(f"  Config: {compile_str}")
        
        # Show local option if available
        local_grad = [r for r in grad_acc if r['gpu'].price_per_hour == 0]
        if local_grad:
            best_local = local_grad[0]
            compile_str = "with torch.compile" if best_local.get('use_compile') else "no compile"
            print(f"\n✓ LOCAL OPTION (Free):")
            print(f"  GPU: {best_local['gpu'].name} - ${best_local['total_cost']:.2f}")
            print(f"  Time: {best_local['training_time_hours']:.1f} hours")
            print(f"  Config: {compile_str}")

if __name__ == '__main__':
    results = find_best_options()
    print_results(results)
    
    # Save to file
    with open('gpu_cost_analysis_results.txt', 'w') as f:
        f.write("GPU Cost Analysis Results\n")
        f.write("="*80 + "\n\n")
        for i, result in enumerate(results[:20], 1):
            f.write(f"{i}. {result['method']}\n")
            if result['method'] == 'Gradient Accumulation':
                gpu = result['gpu']
                compile_str = "with torch.compile" if result.get('use_compile', False) else "no compile"
                f.write(f"   GPU: {gpu.name} ({gpu.vram_gb}GB) - ${gpu.price_per_hour}/hr | {compile_str}\n")
                f.write(f"   Batch: {result['actual_batch_size']} × {result['accumulation_steps']}\n")
            else:
                gpus = result['gpus']
                compile_str = "with torch.compile" if result.get('use_compile', False) else "no compile"
                f.write(f"   GPUs: {result['num_gpus']}× {gpus[0].name} | {compile_str}\n")
                f.write(f"   Batch: {result['batch_size_per_gpu']} per GPU\n")
            f.write(f"   Cost: ${result['total_cost']:.2f} | Time: {result['training_time_hours']:.1f}hrs\n\n")
    
    print(f"\n✓ Results saved to gpu_cost_analysis_results.txt")

