"""
Extended training script for architecture generalization experiments.
Integrates: experiment_config, model_factory, dataset_factory, wandb
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import wandb

from experiment_config import get_all_experiments, validate_experiment
from utils.model_factory import create_model
from utils.dataset_factory import create_dataset
from utils.utils import count_parameters
from losses import normal_loss, flip_inverted_loss
from losses_optimized import flip_all_loss_vectorized
from hyperparameters import get_hyperparameters, get_optimizer

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may be slower)
    torch.backends.cudnn.deterministic = False  # Set to True for full determinism
    torch.backends.cudnn.benchmark = True  # Faster, but not deterministic


def setup_logging(log_dir='logs'):
    """Setup comprehensive logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_extended_{timestamp}.log')
    
    logger = logging.getLogger('ExtendedTraining')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file


class ExtendedExperimentRunner:
    """Extended experiment runner for architecture generalization with wandb."""
    
    def __init__(self, num_epochs=300, batch_size=128, learning_rate=0.001, 
                 device=None, logger=None, use_amp=True, seed=42, early_stop_patience=50):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger if logger else logging.getLogger('ExtendedTraining')
        self.use_amp = use_amp and torch.cuda.is_available()
        self.early_stop_patience = early_stop_patience
        # Use new API to avoid deprecation warning
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Set seed
        set_seed(seed)
        
        self.logger.info("="*80)
        self.logger.info("EXTENDED EXPERIMENT RUNNER INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        self.logger.info(f"Mixed Precision (AMP): {self.use_amp}")
        self.logger.info(f"Early stopping patience: {early_stop_patience}")
        self.logger.info(f"Seed: {seed}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
    
    def get_model_name(self, exp_config):
        """Convert experiment config to model name for factory."""
        arch = exp_config['architecture']
        fusion = exp_config['fusion_type']
        
        if fusion == 'baseline':
            return f"{arch}_baseline"
        elif fusion == 'early':
            return f"{arch}_flip_early"
        elif fusion == 'late':
            return f"{arch}_flip_late"
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
    
    def train_epoch(self, model, train_loader, optimizer, exp_config):
        """Train for one epoch."""
        model.train()
        total_loss = torch.tensor(0.0, device=self.device)  # Keep as tensor for efficiency
        correct = 0
        total = 0
        flip_mode = exp_config['flip_mode']
        fusion_type = exp_config['fusion_type']
        
        for batch_idx, batch in enumerate(train_loader):
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if fusion_type == 'baseline':
                    images, labels = batch[0], batch[1]
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = model(images)
                    loss = normal_loss(outputs, labels)
                else:
                    images, labels, flips = batch[0], batch[1], batch[2]
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    flips = flips.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if fusion_type in ['early', 'late']:
                        outputs = model(images, flips)
                    else:
                        outputs = model(images)
                    
                    # Compute loss based on flip value
                    normal_mask = (flips == 0)
                    flip_mask = (flips == 1)
                    losses = []
                    
                    if normal_mask.sum() > 0:
                        losses.append(normal_loss(outputs[normal_mask], labels[normal_mask]))
                    
                    if flip_mask.sum() > 0:
                        if flip_mode == 'all':
                            losses.append(flip_all_loss_vectorized(outputs[flip_mask], labels[flip_mask]))
                        elif flip_mode == 'inverted':
                            losses.append(flip_inverted_loss(outputs[flip_mask], labels[flip_mask]))
                    
                    loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=self.device)
            
            # Backward pass
            optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss  # Keep as tensor, no .item() call
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Only call .item() once at the end
        avg_loss = (total_loss / len(train_loader)).item()
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def evaluate(self, model, data_loader, exp_config):
        """Evaluate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        fusion_type = exp_config['fusion_type']
        
        with torch.no_grad():
            for batch in data_loader:
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if fusion_type == 'baseline':
                        images, labels = batch[0], batch[1]
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        outputs = model(images)
                        loss = normal_loss(outputs, labels)
                    else:
                        # Handle both cases: batch with flip (train) or without (val/test)
                        if len(batch) == 3:
                            images, labels, flips = batch[0], batch[1], batch[2]
                        else:
                            # Val/test loaders don't have flip, create dummy zeros (all normal samples)
                            images, labels = batch[0], batch[1]
                            flips = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
                        
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        flips = flips.to(self.device, non_blocking=True)
                        
                        if fusion_type in ['early', 'late']:
                            outputs = model(images, flips)
                        else:
                            outputs = model(images)
                        
                        # For evaluation, only use normal samples (flip=0)
                        normal_mask = (flips == 0)
                        if normal_mask.sum() > 0:
                            loss = normal_loss(outputs[normal_mask], labels[normal_mask])
                        else:
                            loss = torch.tensor(0.0, device=self.device)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def run_experiment(self, exp_config, exp_idx=None, total_experiments=None):
        """Run a single experiment with wandb logging."""
        # Generate experiment name for wandb
        exp_name = f"{exp_config['architecture']}_{exp_config['fusion_type']}_{exp_config['flip_mode']}"
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"EXPERIMENT: {exp_name}")
        self.logger.info("="*80)
        
        # Validate experiment
        if not validate_experiment(exp_config):
            self.logger.error(f"Invalid experiment config: {exp_config}")
            return None
        
        # Get hyperparameters FIRST to use paper-matching values
        hyperparams = get_hyperparameters(exp_config['architecture'])
        paper_batch_size = hyperparams['batch_size']
        paper_lr = hyperparams['learning_rate']
        
        # Check for existing checkpoint to resume WandB run
        checkpoints_dir = os.path.join("checkpoints", exp_name)
        latest_checkpoint = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        wandb_run_id = None
        
        if os.path.exists(latest_checkpoint):
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                wandb_run_id = checkpoint.get('wandb_run_id', None)
                if wandb_run_id:
                    self.logger.info(f"Found checkpoint with WandB run ID: {wandb_run_id}")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint for WandB resume: {e}")
        
        # Initialize wandb run with paper-matching hyperparameters
        if wandb_run_id:
            wandb.init(
                project="flip-data-tinyimagenet",
                name=exp_name,
                id=wandb_run_id,
                resume="allow",
                config={
                    'architecture': exp_config['architecture'],
                    'dataset': exp_config['dataset'],
                    'fusion_type': exp_config['fusion_type'],
                    'flip_mode': exp_config['flip_mode'],
                    'use_augmentation': exp_config['use_augmentation'],
                    'num_classes': exp_config['num_classes'],
                    'image_size': exp_config['image_size'],
                    'batch_size': paper_batch_size,  # Use paper-matching batch size
                    'learning_rate': paper_lr,       # Use paper-matching learning rate
                    'num_epochs': self.num_epochs,
                    'early_stop_patience': self.early_stop_patience,
                    'seed': 42,
                }
            )
            self.logger.info(f"Resuming WandB run: {wandb_run_id}")
        else:
            wandb.init(
                project="flip-data-tinyimagenet",
                name=exp_name,
                config={
                    'architecture': exp_config['architecture'],
                    'dataset': exp_config['dataset'],
                    'fusion_type': exp_config['fusion_type'],
                    'flip_mode': exp_config['flip_mode'],
                    'use_augmentation': exp_config['use_augmentation'],
                    'num_classes': exp_config['num_classes'],
                    'image_size': exp_config['image_size'],
                    'batch_size': paper_batch_size,  # Use paper-matching batch size
                    'learning_rate': paper_lr,       # Use paper-matching learning rate
                    'num_epochs': self.num_epochs,
                    'early_stop_patience': self.early_stop_patience,
                    'seed': 42,
                }
            )
        
        # Create data loaders
        self.logger.info("Creating data loaders...")
        try:
            train_loader, val_loader, test_loader = create_dataset(
                dataset_name=exp_config['dataset'],
                batch_size=paper_batch_size,  # Use paper-matching batch size
                use_augmentation=exp_config['use_augmentation'],
                use_flip=(exp_config['fusion_type'] != 'baseline')
            )
            self.logger.info(f"✓ Data loaders created (train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)})")
        except Exception as e:
            self.logger.error(f"✗ Failed to create data loaders: {e}", exc_info=True)
            return None
        
        # Create model
        self.logger.info("Creating model...")
        try:
            model_name = self.get_model_name(exp_config)
            model = create_model(
                model_name=model_name,
                num_classes=exp_config['num_classes']
            ).to(self.device)
            num_params = count_parameters(model)
            
            # Try to compile model for speedup (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info("✓ Model compiled with torch.compile (PyTorch 2.0+)")
                else:
                    self.logger.info("✓ Model created (torch.compile not available)")
            except Exception as compile_error:
                self.logger.warning(f"Model compilation failed, using standard model: {compile_error}")
            
            self.logger.info(f"✓ Model created: {model_name} ({num_params:,} parameters)")
        except Exception as e:
            self.logger.error(f"✗ Failed to create model: {e}", exc_info=True)
            return None
        
        # Get hyperparameters (optimizer, LR, weight decay, scheduler) for this architecture
        # Note: hyperparams already retrieved above for batch_size
        weight_decay = hyperparams.get('weight_decay', 1e-4)

        # 1) Optimizer matching the paper
        optimizer = get_optimizer(model, hyperparams)

        # 2) Scheduler matching the paper
        scheduler_name = hyperparams.get('scheduler', None)
        scheduler_params = hyperparams.get('scheduler_params', {})

        if scheduler_name == 'step':
            milestones = scheduler_params.get('milestones', [])
            gamma = scheduler_params.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosine_warmup':
            import math
            warmup_epochs = scheduler_params.get('warmup_epochs', 10)

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                progress = float(epoch - warmup_epochs) / float(max(1, self.num_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_name == 'rmsprop_decay':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        else:
            scheduler = None
        # Log hyperparameters being used (paper-matching values)
        optimizer_name = hyperparams.get('optimizer', 'adam')
        paper_lr = hyperparams.get('learning_rate', 0.001)
        scheduler_desc = f"{scheduler_name}"
        if scheduler_name == 'step':
            milestones = scheduler_params.get('milestones', [])
            scheduler_desc = f"MultiStepLR (milestones={milestones}, gamma={scheduler_params.get('gamma', 0.1)})"
        elif scheduler_name == 'cosine_warmup':
            warmup = scheduler_params.get('warmup_epochs', 10)
            scheduler_desc = f"CosineWarmup (warmup={warmup} epochs)"
        elif scheduler_name == 'rmsprop_decay':
            scheduler_desc = "ExponentialLR (gamma=0.97)"
        elif scheduler_name is None:
            scheduler_desc = "None (constant LR)"
        
        self.logger.info(f"Training hyperparameters (paper-matching):")
        self.logger.info(f"  Architecture: {exp_config['architecture']}")
        self.logger.info(f"  Batch size: {paper_batch_size} (from paper config)")
        self.logger.info(f"  Optimizer: {optimizer_name.upper()}")
        self.logger.info(f"  Learning rate: {paper_lr:.6f} (initial, from paper)")
        self.logger.info(f"  Weight decay: {weight_decay:.0e}")
        self.logger.info(f"  LR Scheduler: {scheduler_desc}")
        
        # Check for existing checkpoint to resume training
        checkpoints_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        latest_checkpoint = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        
        start_epoch = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': []
        }
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_epoch = 0
        last_improvement_epoch = 0  # Track actual epoch of last improvement
        
        if os.path.exists(latest_checkpoint):
            try:
                self.logger.info(f"Loading checkpoint from: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state (LambdaLR doesn't have state_dict, skip it)
                if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except (AttributeError, TypeError):
                        # LambdaLR and some custom schedulers don't support state_dict
                        # The scheduler will continue from the current epoch
                        self.logger.warning("Scheduler state_dict not supported, continuing from current epoch")
                
                # Load scaler state (AMP)
                if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                # Load training state
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                best_test_acc = checkpoint.get('best_test_acc', 0.0)
                best_epoch = checkpoint.get('best_epoch', 0)
                # Support both old (epochs_without_improvement) and new (last_improvement_epoch) format
                if 'last_improvement_epoch' in checkpoint:
                    last_improvement_epoch = checkpoint.get('last_improvement_epoch', 0)
                else:
                    last_improvement_epoch = best_epoch if best_epoch > 0 else 0
                history = checkpoint.get('history', history)
                
                # Restore start_time if available
                if 'start_time' in checkpoint and checkpoint['start_time']:
                    start_time = datetime.fromisoformat(checkpoint['start_time'])
                else:
                    start_time = datetime.now()
                
                self.logger.info(f"✓ Resumed from epoch {start_epoch}")
                self.logger.info(f"  Best val acc: {best_val_acc:.2f}% (epoch {best_epoch})")
                epochs_since_improvement = start_epoch - last_improvement_epoch if last_improvement_epoch > 0 else 0
                self.logger.info(f"  Epochs since last improvement: {epochs_since_improvement}")
                
                # Step scheduler to correct epoch (for schedulers that track step count)
                # LambdaLR recalculates based on epoch number, so this is mainly for MultiStepLR/ExponentialLR
                # We step start_epoch-1 times so that after the first training epoch, it's at start_epoch
                if scheduler is not None and start_epoch > 0:
                    for _ in range(start_epoch - 1):
                        scheduler.step()
                    self.logger.info(f"  Scheduler stepped to epoch {start_epoch - 1} (will be at {start_epoch} after first training step)")
            except Exception as e:
                self.logger.error(f"✗ Failed to load checkpoint: {e}", exc_info=True)
                self.logger.info("Starting training from scratch...")
                start_time = datetime.now()
        else:
            self.logger.info("No checkpoint found, starting fresh training...")
            start_time = datetime.now()
        
        # Training loop with early stopping
        for epoch in range(start_epoch, self.num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, exp_config)

            if scheduler is not None:
                scheduler.step()
                # Some schedulers (LambdaLR/MultiStepLR) use optimizer.param_groups directly,
                # but get_last_lr() works for all standard schedulers.
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate every 5 epochs, or on first/last epoch
            should_evaluate = (epoch == 0) or (epoch + 1) % 5 == 0 or (epoch + 1 == self.num_epochs)
            
            if should_evaluate:
                val_loss, val_acc = self.evaluate(model, val_loader, exp_config)
                test_loss, test_acc = self.evaluate(model, test_loader, exp_config)
                
                # Update history
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
            else:
                # Use last known values for logging (or None if first epoch)
                if len(history['val_loss']) > 0:
                    val_loss = history['val_loss'][-1]
                    val_acc = history['val_acc'][-1]
                    test_loss = history['test_loss'][-1]
                    test_acc = history['test_acc'][-1]
                else:
                    # First epoch, evaluate to get baseline
                    val_loss, val_acc = self.evaluate(model, val_loader, exp_config)
                    test_loss, test_acc = self.evaluate(model, test_loader, exp_config)
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)
            
            # Always update train history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Log to wandb (only log evaluation metrics when we actually evaluate)
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'learning_rate': current_lr,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
            }
            if should_evaluate:
                log_dict.update({
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                })
            wandb.log(log_dict)
            
            # Track best and early stopping (only when we actually evaluate)
            if should_evaluate:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                    last_improvement_epoch = epoch + 1  # Update actual epoch of last improvement
                    # Save best model locally
                    model_path = os.path.join(checkpoints_dir, f"best_model_epoch_{best_epoch}.pth")
                    torch.save(model.state_dict(), model_path)
                    #wandb.save(model_path)
            
            # Save checkpoint every 5 epochs and always save latest checkpoint
            save_periodic = (epoch + 1) % 5 == 0
            save_latest = True  # Always save latest for immediate resume
            
            if save_periodic or save_latest:
                # Try to get scheduler state_dict (LambdaLR doesn't support it)
                scheduler_state = None
                if scheduler is not None:
                    try:
                        scheduler_state = scheduler.state_dict()
                    except (AttributeError, TypeError):
                        # LambdaLR and custom schedulers don't have state_dict
                        scheduler_state = None
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state,
                    'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                    'best_epoch': best_epoch,
                    'last_improvement_epoch': last_improvement_epoch,
                    'history': history,
                    'wandb_run_id': wandb.run.id if wandb.run else None,
                    'start_time': start_time.isoformat(),
                }
                
                # Save latest checkpoint (for immediate resume)
                if save_latest:
                    latest_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
                    torch.save(checkpoint, latest_path)
                    #wandb.save(latest_path)
                
                # Save periodic checkpoint (every 5 epochs)
                if save_periodic:
                    periodic_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                    torch.save(checkpoint, periodic_path)
                    #wandb.save(periodic_path)
                    self.logger.info(f"Checkpoint saved: epoch {epoch + 1}")
            
            # Log progress (every 10 epochs or when we evaluate)
            if (epoch + 1) % 10 == 0 or epoch == 0 or should_evaluate:
                if should_evaluate:
                    # Calculate epochs since last improvement
                    if last_improvement_epoch > 0:
                        # We've seen improvement, count from that epoch
                        epochs_since_improvement = (epoch + 1) - last_improvement_epoch
                    else:
                        # Never seen improvement, count from start
                        epochs_since_improvement = (epoch + 1) - start_epoch
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                        f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
                        f"Test: {test_loss:.4f}/{test_acc:.2f}% | "
                        f"Best: {best_val_acc:.2f}% (epoch {best_epoch}) | "
                        f"LR: {current_lr:.6f} | "
                        f"No improvement: {epochs_since_improvement}/{self.early_stop_patience} epochs"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                        f"(Skipped eval) | "
                        f"Best: {best_val_acc:.2f}% (epoch {best_epoch}) | "
                        f"LR: {current_lr:.6f}"
                    )
            
            # Early stopping (check actual epochs since last improvement)
            if last_improvement_epoch > 0:
                # We've seen improvement, count from that epoch
                epochs_since_improvement = (epoch + 1) - last_improvement_epoch
            else:
                # Never seen improvement, count from start
                epochs_since_improvement = (epoch + 1) - start_epoch
            
            if epochs_since_improvement >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {epochs_since_improvement} epochs)")
                wandb.log({'early_stopped': True, 'stopped_at_epoch': epoch + 1})
                break
        
        # Ensure we evaluate on the final epoch if training completed normally
        if len(history['val_loss']) == 0 or (len(history['train_loss']) > 0 and len(history['val_loss']) < len(history['train_loss'])):
            # Final epoch wasn't evaluated, evaluate now
            self.logger.info("Evaluating final model...")
            final_val_loss, final_val_acc = self.evaluate(model, val_loader, exp_config)
            final_test_loss, final_test_acc = self.evaluate(model, test_loader, exp_config)
            history['val_loss'].append(final_val_loss)
            history['val_acc'].append(final_val_acc)
            history['test_loss'].append(final_test_loss)
            history['test_acc'].append(final_test_acc)
            
            # Update best if final is better
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_test_acc = final_test_acc
                best_epoch = len(history['train_loss'])
        
        # Get final test accuracy from history
        final_test_acc = history['test_acc'][-1] if len(history['test_acc']) > 0 else 0.0
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        actual_epochs = len(history['train_loss'])
        
        # Log final metrics to wandb
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'best_epoch': best_epoch,
            'total_training_time_seconds': training_time,
            'actual_epochs': actual_epochs,
            'num_parameters': num_params,
        })
        
        # Summary metrics for wandb
        wandb.summary.update({
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'best_epoch': best_epoch,
            'total_training_time_seconds': training_time,
            'actual_epochs': actual_epochs,
            'num_parameters': num_params,
        })
        
        self.logger.info(f"✓ Training completed: Best val acc: {best_val_acc:.2f}%, Best test acc: {best_test_acc:.2f}%")
        self.logger.info(f"  Trained for {actual_epochs} epochs (out of {self.num_epochs} max)")
        self.logger.info(f"  Training time: {training_time/3600:.2f} hours")
        
        # Save final checkpoint
        # Try to get scheduler state_dict (LambdaLR doesn't support it)
        scheduler_state = None
        if scheduler is not None:
            try:
                scheduler_state = scheduler.state_dict()
            except (AttributeError, TypeError):
                # LambdaLR and custom schedulers don't have state_dict
                scheduler_state = None
        
        final_checkpoint = {
            'epoch': actual_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'last_improvement_epoch': last_improvement_epoch,
            'history': history,
            'wandb_run_id': wandb.run.id if wandb.run else None,
            'start_time': start_time.isoformat(),
        }
        final_checkpoint_path = os.path.join(checkpoints_dir, "final_checkpoint.pth")
        torch.save(final_checkpoint, final_checkpoint_path)
        #wandb.save(final_checkpoint_path)
        
        # Finish wandb run
        wandb.finish()
        
        return {
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'best_epoch': best_epoch,
            'actual_epochs': actual_epochs,
        }
    
    def run_all_experiments(self, filter_config=None):
        """Run all experiments."""
        all_experiments = get_all_experiments()
        
        if filter_config:
            from experiment_config import filter_experiments
            all_experiments = filter_experiments(all_experiments, **filter_config)
        
        total = len(all_experiments)
        self.logger.info(f"Total experiments to run: {total}")
        
        results = []
        for idx, exp_config in enumerate(all_experiments, 1):
            result = self.run_experiment(exp_config, idx, total)
            if result:
                results.append(result)
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"ALL EXPERIMENTS COMPLETED: {len(results)}/{total} successful")
        self.logger.info("="*80)
        
        return results


if __name__ == '__main__':
    # Setup
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("EXTENDED ARCHITECTURE GENERALIZATION EXPERIMENTS")
    logger.info("="*80)
    
    # Configuration
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    USE_AMP = True
    SEED = 42
    
    logger.info(f"Configuration: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")
    
    # Create runner
    runner = ExtendedExperimentRunner(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logger=logger,
        use_amp=USE_AMP,
        seed=SEED
    )
    
    # Run experiments
    # You can filter experiments here, e.g.:
    # filter_config = {'architecture': 'resnet18'}  # Only ResNet-18
    filter_config = None  # Run all experiments
    
    try:
        results = runner.run_all_experiments(filter_config=filter_config)
        logger.info("="*80)
        logger.info("SUCCESS!")
        logger.info("="*80)
    except Exception as e:
        logger.error("="*80)
        logger.error("FAILED!", exc_info=True)
        logger.error("="*80)
        raise



