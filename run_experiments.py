"""
Script to run experiments with wandb integration.
Supports both direct execution and wandb sweep agent mode.
"""
import argparse
import logging
import wandb
from train_extended import ExtendedExperimentRunner, setup_logging
from experiment_config import get_all_experiments, filter_experiments, validate_experiment
from hyperparameters import get_hyperparameters
import torch
import os


def main():
    parser = argparse.ArgumentParser(description='Run flip data experiments')
    parser.add_argument('--architecture', type=str, default=None,
                       help='Filter by architecture (e.g., resnet18)')
    parser.add_argument('--fusion_type', type=str, default=None,
                       help='Filter by fusion type (baseline, early, late)')
    parser.add_argument('--flip_mode', type=str, default=None,
                       help='Filter by flip mode (none, all, inverted)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("STARTING EXPERIMENT RUN")
    logger.info("="*80)
    
    # Check if running as wandb sweep agent
    sweep_id = os.environ.get('WANDB_SWEEP_ID')
    if sweep_id:
        # Running as sweep agent - get config from wandb
        wandb.init()
        config = wandb.config
        
        exp_config = {
            'architecture': config.architecture,
            'dataset': 'tinyimagenet',
            'fusion_type': config.fusion_type,
            'flip_mode': config.flip_mode,
            'use_augmentation': True,  # Always True
            'num_classes': 200,
            'image_size': 64,
        }
        
        # Validate experiment
        if not validate_experiment(exp_config):
            logger.error(f"Invalid experiment config: {exp_config}")
            return
        
        # Get architecture-specific hyperparameters
        arch = exp_config['architecture']
        hyperparams = get_hyperparameters(arch)
        
        # Override with command-line args if provided
        if args.batch_size:
            hyperparams['batch_size'] = args.batch_size
        if args.learning_rate:
            hyperparams['learning_rate'] = args.learning_rate
        
        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment runner
        runner = ExtendedExperimentRunner(
            num_epochs=args.num_epochs,
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            device=device,
            logger=logger,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience
        )
        
        # Run single experiment
        result = runner.run_experiment(exp_config)
        
        if result:
            logger.info("✓ Experiment completed successfully")
        else:
            logger.error("✗ Experiment failed")
            exit(1)
    else:
        # Running directly - get experiments from config
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
        
        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")
        
        # Run experiments
        completed = 0
        failed = 0
        
        for idx, exp_config in enumerate(valid_experiments):
            try:
                # Get architecture-specific hyperparameters
                arch = exp_config['architecture']
                hyperparams = get_hyperparameters(arch)
                
                # Override with command-line args if provided
                if args.batch_size:
                    hyperparams['batch_size'] = args.batch_size
                if args.learning_rate:
                    hyperparams['learning_rate'] = args.learning_rate
                
                # Create experiment runner
                runner = ExtendedExperimentRunner(
                    num_epochs=args.num_epochs,
                    batch_size=hyperparams['batch_size'],
                    learning_rate=hyperparams['learning_rate'],
                    device=device,
                    logger=logger,
                    seed=args.seed,
                    early_stop_patience=args.early_stop_patience
                )
                
                # Run experiment
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

