import argparse
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from train_extended import ExtendedExperimentRunner, setup_logging
from experiment_config import get_all_experiments, filter_experiments, validate_experiment
from hyperparameters import get_hyperparameters


def setup_ddp(rank, world_size, backend: str = 'nccl'):
    """Initialize distributed training for a single worker process."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_worker_ddp(rank, world_size, args, exp_config):
    """Run training on a single GPU (worker process) for DDP."""
    logger = None
    try:
        setup_ddp(rank, world_size)

        # Setup logging (only rank 0 logs to console in detail)
        if rank == 0:
            logger, log_file = setup_logging()
            logger.info(f"Starting distributed training on {world_size} GPUs")
        else:
            logger = logging.getLogger(f'Worker_{rank}')
            logger.setLevel(logging.WARNING)

        # Get hyperparameters with num_gpus for auto-adjustment
        num_gpus = world_size
        hyperparams = get_hyperparameters(exp_config['architecture'], num_gpus=num_gpus)

        # Collect extra optimizer params (e.g. RMSProp alpha/momentum/eps)
        extra_opt_params = {}
        for key in ('alpha', 'momentum', 'eps'):
            if key in hyperparams:
                extra_opt_params[key] = hyperparams[key]

        runner = ExtendedExperimentRunner(
            num_epochs=args.num_epochs,
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            device=None,  # Device is set per-rank inside ExtendedExperimentRunner when use_ddp=True
            logger=logger,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            use_ddp=True,
            rank=rank,
            world_size=world_size,
            optimizer=hyperparams.get('optimizer'),
            weight_decay=hyperparams.get('weight_decay'),
            effective_batch_size=hyperparams.get('effective_batch_size'),
            gradient_clip=hyperparams.get('gradient_clip'),
            scheduler=hyperparams.get('scheduler'),
            scheduler_params=hyperparams.get('scheduler_params', {}),
            optimizer_extra_params=extra_opt_params,
        )

        # Run experiment (WandB initialization happens inside run_experiment)
        result = runner.run_experiment(exp_config)

        cleanup_ddp()

        if rank == 0:
            if result:
                logger.info("✓ Distributed training completed successfully")
            else:
                logger.error("✗ Distributed training failed")

        return result

    except Exception as e:
        if logger is not None and rank == 0:
            logger.error(f"Error in worker {rank}: {e}", exc_info=True)
        cleanup_ddp()
        raise


def launch_single(args, exp_config, device, logger):
    """Launch a single-process (single-GPU/CPU) experiment run."""
    # Get hyperparameters (single GPU, so num_gpus=1)
    hyperparams = get_hyperparameters(exp_config['architecture'], num_gpus=1)
    
    # Collect extra optimizer params (e.g. RMSProp alpha/momentum/eps)
    extra_opt_params = {}
    for key in ('alpha', 'momentum', 'eps'):
        if key in hyperparams:
            extra_opt_params[key] = hyperparams[key]
    
    runner = ExtendedExperimentRunner(
        num_epochs=args.num_epochs,
        batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        device=device,
        logger=logger,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        use_ddp=False,
        rank=0,
        world_size=1,
        optimizer=hyperparams.get('optimizer'),
        weight_decay=hyperparams.get('weight_decay'),
        effective_batch_size=hyperparams.get('effective_batch_size'),
        gradient_clip=hyperparams.get('gradient_clip'),
        scheduler=hyperparams.get('scheduler'),
        scheduler_params=hyperparams.get('scheduler_params', {}),
        optimizer_extra_params=extra_opt_params,
    )

    return runner.run_experiment(exp_config)


def launch_ddp(args, exp_config, logger):
    """Launch a multi-GPU DDP run using mp.spawn."""
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.error("DDP requested but less than 2 GPUs are available.")
        return None

    logger.info(f"Launching DDP on {world_size} GPUs")
    mp.spawn(
        run_worker_ddp,
        args=(world_size, args, exp_config),
        nprocs=world_size,
        join=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Run flip data experiments')
    parser.add_argument('--architecture', type=str, default=None,
                       help='Filter by architecture (e.g., resnet18) or required for sweep')
    parser.add_argument('--fusion_type', type=str, default=None,
                       help='Filter by fusion type (baseline, early, late) or required for sweep')
    parser.add_argument('--flip_mode', type=str, default=None,
                       help='Filter by flip mode (none, all, inverted) or required for sweep')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, default=1500,
                       help='Number of training epochs (default: 1500)')
    parser.add_argument('--early_stop_patience', type=int, default=200,
                       help='Early stopping patience (default: 150)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING EXPERIMENT RUN")
    logger.info("=" * 80)

    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Check if running a single, fully specified experiment
    if args.architecture and args.fusion_type and args.flip_mode:
        # Running as sweep agent or single experiment - run one experiment
        exp_config = {
            'architecture': args.architecture,
            'dataset': 'tinyimagenet',
            'fusion_type': args.fusion_type,
            'flip_mode': args.flip_mode,
            'use_augmentation': True,  # Always True
            'num_classes': 200,
            'image_size': 64,
        }

        # Validate experiment
        if not validate_experiment(exp_config):
            logger.error(f"Invalid experiment config: {exp_config}")
            return

        # Determine device for single-process run
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if architecture allows DDP
        hyperparams = get_hyperparameters(exp_config['architecture'], num_gpus=num_gpus)
        allow_ddp = hyperparams.get('allow_ddp', False)
        
        # Decide between single-process and DDP based on allow_ddp flag
        if num_gpus > 1 and allow_ddp:
            logger.info(f"Architecture '{exp_config['architecture']}' allows DDP. Multiple GPUs detected ({num_gpus}); launching DDP run.")
            if hyperparams.get('_auto_adjusted_batch', False):
                logger.info(f"Auto-adjusted batch_size: {hyperparams['_original_batch']} → {hyperparams['batch_size']} per GPU "
                          f"(effective_batch_size: {hyperparams.get('effective_batch_size', 'N/A')} across {num_gpus} GPUs)")
            result = launch_ddp(args, exp_config, logger)
        else:
            if num_gpus > 1 and not allow_ddp:
                logger.info(f"Architecture '{exp_config['architecture']}' is configured for single-GPU only. "
                          f"Ignoring {num_gpus} available GPUs and using single-GPU mode.")
            result = launch_single(args, exp_config, device, logger)
        
        if result:
            logger.info("✓ Experiment completed successfully")
        else:
            logger.error("✗ Experiment failed")
            exit(1)
    else:
        # Running directly without all args - get experiments from config
        all_experiments = get_all_experiments()
        
        # Apply filters
        if args.architecture:
            all_experiments = filter_experiments(all_experiments, architecture=args.architecture)
        if args.fusion_type:
            all_experiments = filter_experiments(all_experiments, fusion_type=args.fusion_type)
        if args.flip_mode:
            all_experiments = filter_experiments(all_experiments, flip_mode=args.flip_mode)
        
        # Validate all experiments
        valid_experiments = [exp for exp in all_experiments if validate_experiment(exp)]
        logger.info(f"Total experiments to run: {len(valid_experiments)}")
        
        if len(valid_experiments) == 0:
            logger.warning("No valid experiments to run!")
            return
        
        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")

        # For multi-experiment runs, always use single-GPU mode (even if multiple GPUs available)
        # This ensures experiments run sequentially and don't interfere with each other
        if num_gpus > 1:
            logger.info(f"Multi-experiment mode: Using single-GPU mode even though {num_gpus} GPUs are available. "
                       f"Each experiment will run sequentially on a single GPU.")

        # Run experiments (single-process, even if multiple GPUs are available)
        completed = 0
        failed = 0
        
        for idx, exp_config in enumerate(valid_experiments):
            try:
                # Get ALL hyperparameters for this architecture (single GPU mode)
                hyperparams = get_hyperparameters(exp_config['architecture'], num_gpus=1)
                
                # Collect extra optimizer params (e.g. RMSProp alpha/momentum/eps)
                extra_opt_params = {}
                for key in ('alpha', 'momentum', 'eps'):
                    if key in hyperparams:
                        extra_opt_params[key] = hyperparams[key]
                
                # Create experiment runner with all hyperparameters
                runner = ExtendedExperimentRunner(
                    num_epochs=args.num_epochs,
                    batch_size=hyperparams['batch_size'],
                    learning_rate=hyperparams['learning_rate'],
                    device=device,
                    logger=logger,
                    seed=args.seed,
                    early_stop_patience=args.early_stop_patience,
                    use_ddp=False,
                    rank=0,
                    world_size=1,
                    # Pass all hyperparameters
                    optimizer=hyperparams.get('optimizer'),
                    weight_decay=hyperparams.get('weight_decay'),
                    effective_batch_size=hyperparams.get('effective_batch_size'),
                    gradient_clip=hyperparams.get('gradient_clip'),
                    scheduler=hyperparams.get('scheduler'),
                    scheduler_params=hyperparams.get('scheduler_params', {}),
                    optimizer_extra_params=extra_opt_params,
                )

                # Run experiment (WandB initialization happens inside run_experiment)
                result = runner.run_experiment(exp_config, idx + 1, len(valid_experiments))
                
                if result:
                    completed += 1
                    logger.info(f"✓ Experiment {idx + 1}/{len(valid_experiments)} completed")
                else:
                    failed += 1
                    logger.error(f"✗ Experiment {idx + 1}/{len(valid_experiments)} failed")
                    
            except KeyboardInterrupt:
                logger.warning("="*80)
                logger.warning("Interrupted by user (Ctrl+C)")
                logger.warning(f"Progress: {completed} completed, {failed} failed")
                break
            except Exception as e:
                failed += 1
                logger.error(f"✗ Experiment {idx + 1}/{len(valid_experiments)} failed with error: {e}", exc_info=True)
        
        # Summary
        logger.info("="*80)
        logger.info("EXPERIMENT RUN COMPLETE")
        logger.info("="*80)
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {completed + failed}")


if __name__ == '__main__':
    main()
