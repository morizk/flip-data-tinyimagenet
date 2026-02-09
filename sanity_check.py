"""
Sanity check script for all architectures with GPU monitoring.
Runs 1-epoch training on each baseline architecture and monitors:
- GPU memory usage
- GPU utilization
- Training time
- Throughput (samples/second)
"""
import torch
import torch.nn as nn
import time
import psutil
import subprocess
from train_extended import ExtendedExperimentRunner, setup_logging
from experiment_config import get_all_experiments, filter_experiments
from hyperparameters import get_hyperparameters


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9, torch.cuda.max_memory_allocated(0) / 1e9
    return 0, 0


def get_gpu_utilization():
    """Get GPU utilization percentage using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None


def run_sanity_check(architecture, num_epochs=1, batch_size=None):
    """Run sanity check for a specific architecture."""
    print(f"\n{'='*80}")
    print(f"SANITY CHECK: {architecture.upper()}")
    print(f"{'='*80}")
    
    # Get architecture-specific hyperparameters
    hyperparams = get_hyperparameters(architecture)
    if batch_size is None:
        batch_size = hyperparams['batch_size']
    
    # Get experiment config
    all_exps = get_all_experiments()
    baseline_exps = filter_experiments(
        all_exps,
        architecture=architecture,
        fusion_type='baseline',
        use_augmentation=False
    )
    
    if not baseline_exps:
        print(f"✗ No baseline experiment found for {architecture}")
        return None
    
    exp_config = baseline_exps[0]
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Get initial GPU state
    initial_memory, _ = get_gpu_memory()
    initial_util = get_gpu_utilization()
    
    # Create runner
    logger, _ = setup_logging()
    runner = ExtendedExperimentRunner(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=hyperparams['learning_rate'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        seed=42
    )
    
    # Monitor GPU during training
    start_time = time.time()
    start_memory, _ = get_gpu_memory()
    
    print(f"Configuration:")
    print(f"  Architecture: {architecture}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {runner.device}")
    
    # Run experiment
    try:
        result = runner.run_experiment(exp_config, 1, 1)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Get final GPU state
        final_memory, peak_memory = get_gpu_memory()
        final_util = get_gpu_utilization()
        
        # Calculate throughput
        # TinyImageNet: 100,000 train samples, doubled with flip = 200,000
        # But baseline doesn't use flip, so 100,000 samples
        num_samples = 100000
        if exp_config.get('fusion_type') != 'baseline':
            num_samples *= 2  # Flip dataset doubles samples
        
        throughput = num_samples / training_time
        
        print(f"\n{'='*80}")
        print(f"RESULTS:")
        print(f"{'='*80}")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Throughput: {throughput:.0f} samples/second")
        print(f"Estimated time for 100 epochs: {training_time * 100 / 60:.1f} minutes ({training_time * 100 / 3600:.2f} hours)")
        
        if torch.cuda.is_available():
            print(f"\nGPU Memory:")
            print(f"  Initial: {initial_memory:.2f} GB")
            print(f"  Peak: {peak_memory:.2f} GB")
            print(f"  Final: {final_memory:.2f} GB")
            print(f"  Memory used: {peak_memory - initial_memory:.2f} GB")
            
            if initial_util is not None:
                print(f"\nGPU Utilization:")
                print(f"  Initial: {initial_util:.1f}%")
                if final_util is not None:
                    print(f"  Final: {final_util:.1f}%")
        
        if result and 'metrics' in result:
            metrics = result['metrics']
            print(f"\nTraining Metrics:")
            best_val_acc = metrics.get('best_val_acc', None)
            best_test_acc = metrics.get('best_test_acc', None)
            best_epoch = metrics.get('best_epoch', None)
            if best_val_acc is not None:
                print(f"  Best val acc: {best_val_acc:.2f}%")
            if best_test_acc is not None:
                print(f"  Best test acc: {best_test_acc:.2f}%")
            if best_epoch is not None:
                print(f"  Best epoch: {best_epoch}")
        
        print(f"{'='*80}")
        
        return {
            'architecture': architecture,
            'training_time': training_time,
            'throughput': throughput,
            'peak_memory_gb': peak_memory,
            'memory_used_gb': peak_memory - initial_memory,
            'estimated_100_epochs_minutes': training_time * 100 / 60,
            'success': True
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'architecture': architecture,
            'success': False,
            'error': str(e)
        }


def main():
    """Run sanity checks for all architectures."""
    print("="*80)
    print("ARCHITECTURE SANITY CHECKS WITH GPU MONITORING")
    print("="*80)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\n⚠ No GPU available, will use CPU (will be slower)")
    
    # Architectures to test
    architectures = ['resnet18', 'vgg11', 'efficientnetb0', 'vit']
    
    # Adjust batch sizes for GPU memory constraints (6GB GPU)
    batch_sizes = {
        'resnet18': 64,  # Reduced from 128 for 6GB GPU
        'vgg11': 32,     # Reduced from 64 for 6GB GPU
        'efficientnetb0': 64,  # Reduced from 128
        'vit': 16,  # Reduced from 64 for 6GB GPU
    }
    
    results = []
    
    for arch in architectures:
        batch_size = batch_sizes.get(arch)
        result = run_sanity_check(arch, num_epochs=1, batch_size=batch_size)
        if result:
            results.append(result)
        
        # Clear cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nSuccessful: {len(successful)}/{len(architectures)}")
    print(f"Failed: {len(failed)}/{len(architectures)}")
    
    if successful:
        print(f"\nPerformance Summary:")
        print(f"{'Architecture':<20} {'Time (min)':<15} {'Est. 100E (hrs)':<18} {'Peak Mem (GB)':<15} {'Throughput':<15}")
        print("-"*80)
        for r in successful:
            print(f"{r['architecture']:<20} {r['training_time']/60:<15.2f} {r['estimated_100_epochs_minutes']/60:<18.2f} {r['peak_memory_gb']:<15.2f} {r['throughput']:<15.0f}")
        
        # Find slowest/fastest
        if successful:
            slowest = max(successful, key=lambda x: x['training_time'])
            fastest = min(successful, key=lambda x: x['training_time'])
            print(f"\nFastest: {fastest['architecture']} ({fastest['training_time']/60:.2f} min)")
            print(f"Slowest: {slowest['architecture']} ({slowest['training_time']/60:.2f} min)")
    
    if failed:
        print(f"\nFailed architectures:")
        for r in failed:
            print(f"  - {r['architecture']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if successful:
        max_time = max(r['estimated_100_epochs_minutes']/60 for r in successful)
        total_time = sum(r['estimated_100_epochs_minutes']/60 for r in successful) * 10  # 10 experiments per arch
        print(f"Estimated total time for all 60 experiments: {total_time:.1f} hours ({total_time/24:.1f} days)")
        print(f"Longest single experiment: {max_time:.1f} hours")
        
        max_memory = max(r['peak_memory_gb'] for r in successful)
        print(f"Peak GPU memory usage: {max_memory:.2f} GB")
        if max_memory > 5.5:
            print("⚠ WARNING: High memory usage, consider reducing batch sizes further")
    
    print("="*80)


if __name__ == '__main__':
    main()

