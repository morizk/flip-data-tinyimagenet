"""
Extended training script for architecture generalization experiments.
Integrates: experiment_config, model_factory, dataset_factory, wandb
"""
import torch
import torch.optim as optim
import numpy as np
import random
import os
import csv
import logging
from datetime import datetime
import wandb

from experiment_config import get_all_experiments, validate_experiment
from utils.model_factory import create_model
from utils.dataset_factory import create_dataset
from utils.utils import count_parameters
from losses import normal_loss, flip_inverted_loss
from losses_optimized import flip_all_loss_vectorized
from hyperparameters import get_hyperparameters, get_optimizer

# Constants
EVAL_INTERVAL = 5  
LOG_INTERVAL_EPOCHS = 10  
DEFAULT_SEED = 42

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def _load_confusion_topk_targets_for_arch(architecture: str, k: int = 5) -> np.ndarray:
    """
    Load baseline correlation matrix for a given architecture and build
    top‑k confusion targets:
      - For each true class i, take the k most confused *wrong* classes
        from the baseline matrix and assign 1/k probability to each.
      - True class i itself always gets probability 0.

    This uses the precomputed baseline correlation matrices in
    confusion_targets/{architecture}.npy, so no re‑inference is needed.
    """
    arch = architecture.lower()
    # Use organized folder structure: confusion_targets/{architecture}.npy
    path = os.path.join("confusion_targets", f"{arch}.npy")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline correlation matrix not found for architecture '{architecture}': {path}\n"
            f"Expected path: {os.path.abspath(path)}"
        )

    corr = np.load(path)  # shape (C, C)
    num_classes = corr.shape[0]

    targets = np.zeros_like(corr, dtype=np.float32)
    for i in range(num_classes):
        row = corr[i].copy()
        # Exclude true class itself
        row[i] = -1.0
        # Indices of top‑k confused classes
        topk_idx = np.argsort(row)[-k:]
        targets[i, topk_idx] = 1.0 / float(k)

    return targets


class ExtendedExperimentRunner:
    """Extended experiment runner for architecture generalization with wandb."""
    
    def __init__(self, num_epochs=300, batch_size=128, learning_rate=0.001, 
                 device=None, logger=None, use_amp=True, seed=42, early_stop_patience=50,
                 use_ddp=False, rank=0, world_size=1, ddp_backend='nccl',
                 # Hyperparameters (optional - will be set from get_hyperparameters if None)
                 optimizer=None, weight_decay=None, effective_batch_size=None,
                 gradient_clip=None, scheduler=None, scheduler_params=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.early_stop_patience = early_stop_patience
        
        # Store hyperparameters (optional parameters use None as default)
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.effective_batch_size = effective_batch_size if effective_batch_size is not None else batch_size
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        # Confusion targets for new inverted loss (loaded per-architecture)
        self.confusion_targets = None
        
        if use_ddp:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA")
            # Set device for this process
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(rank)
            self.logger = logger if logger else logging.getLogger(f'ExtendedTraining_Rank{rank}')
        else:
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger = logger if logger else logging.getLogger('ExtendedTraining')
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.early_stop_patience = early_stop_patience
        
        # Use new API to avoid deprecation warning
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        set_seed(seed)
        
        if self._is_main_process():
            self.logger.info("="*80)
            self.logger.info("EXTENDED EXPERIMENT RUNNER INITIALIZED")
            self.logger.info("="*80)
            self.logger.info(f"Device: {self.device}")
            if use_ddp:
                self.logger.info(f"Distributed Training: {world_size} GPUs (rank {rank})")
            self.logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
            self.logger.info(f"Mixed Precision (AMP): {self.use_amp}")
            self.logger.info(f"Early stopping patience: {early_stop_patience}")
            self.logger.info(f"Seed: {seed}")
            if torch.cuda.is_available():
                if use_ddp:
                    self.logger.info(f"CUDA: {torch.cuda.get_device_name(rank)} (rank {rank})")
                else:
                    self.logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
                torch.backends.cudnn.benchmark = True
    
    def _is_main_process(self):
        """Check if this is the main process (rank 0 or not using DDP)."""
        return self.rank == 0 or not self.use_ddp
    
    def _optimizer_step(self, optimizer, model, gradient_clip=None):
        """Perform optimizer step with optional gradient clipping.
        
        Args:
            optimizer: Optimizer instance
            model: Model instance
            gradient_clip: Optional gradient clipping value (global norm)
        """
        if gradient_clip is not None:
            if self.use_amp:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        else:
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
        optimizer.zero_grad()
    
    def _get_wandb_config(self, exp_config, paper_batch_size, paper_lr, effective_batch_size):
        """Create WandB config dictionary.
        
        Args:
            exp_config: Experiment configuration
            paper_batch_size: Batch size from hyperparameters
            paper_lr: Learning rate from hyperparameters
            effective_batch_size: Effective batch size (with gradient accumulation)
        
        Returns:
            Dictionary of WandB config values
        """
        return {
            'architecture': exp_config['architecture'],
            'dataset': exp_config['dataset'],
            'fusion_type': exp_config['fusion_type'],
            'flip_mode': exp_config['flip_mode'],
            'use_augmentation': exp_config['use_augmentation'],
            'num_classes': exp_config['num_classes'],
            'image_size': exp_config['image_size'],
            'batch_size': paper_batch_size,
            'effective_batch_size': effective_batch_size,
            'learning_rate': paper_lr,
            'num_epochs': self.num_epochs,
            'early_stop_patience': self.early_stop_patience,
            'seed': DEFAULT_SEED,
        }
    
    def _create_checkpoint_dict(self, model, optimizer, scheduler, epoch, best_val_acc,
                                best_test_acc, best_epoch, last_improvement_epoch, history, start_time):
        """Create checkpoint dictionary.
        
        Args:
            model: Model instance (will be unwrapped if DDP)
            optimizer: Optimizer instance
            scheduler: Scheduler instance (may be None)
            epoch: Current epoch number
            best_val_acc: Best validation accuracy
            best_test_acc: Best test accuracy
            best_epoch: Epoch with best validation accuracy
            last_improvement_epoch: Last epoch with improvement
            history: Training history dictionary
            start_time: Training start time
        
        Returns:
            Dictionary containing checkpoint data
        """
        model_to_save = model.module if self.use_ddp else model
        scheduler_state = None
        if scheduler is not None:
            try:
                scheduler_state = scheduler.state_dict()
            except (AttributeError, TypeError):
                scheduler_state = None
        
        return {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
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
    
    def train_epoch(self, model, train_loader, optimizer, exp_config, epoch_num=None):
        """Train for one epoch."""
        import time
        model.train()
        
        # Set epoch for DistributedSampler (important for proper shuffling)
        if self.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_num if epoch_num is not None else 0)
        
        total_loss = torch.tensor(0.0, device=self.device)  
        correct = 0
        total = 0
        flip_mode = exp_config['flip_mode']
        fusion_type = exp_config['fusion_type']
        
        actual_batch_size = exp_config.get('batch_size', None)
        effective_batch_size = exp_config.get('effective_batch_size', None)
        
        if self.use_ddp:
            accumulation_steps = 1
            if self.rank == 0:
                total_effective_batch = actual_batch_size * self.world_size if actual_batch_size else None
                self.logger.debug(f"DDP mode: batch_size={actual_batch_size} per GPU, "
                                f"total effective batch={total_effective_batch} across {self.world_size} GPUs")
        elif effective_batch_size and actual_batch_size and effective_batch_size > actual_batch_size:
            # Single GPU: use gradient accumulation to reach effective_batch_size
            accumulation_steps = effective_batch_size // actual_batch_size
            if accumulation_steps > 1:
                self.logger.debug(f"Gradient accumulation: {accumulation_steps} steps (effective batch: {effective_batch_size})")
        else:
            accumulation_steps = 1
        
        # Progress tracking
        epoch_start_time = time.time()
        total_batches = len(train_loader)
        # Log more frequently for gradient accumulation (every accumulation step or every 10% of epoch)
        log_interval = max(1, total_batches // 10)  # Log ~10 times per epoch
        if accumulation_steps > 1:
            # Log at least every accumulation step, but not more than every 5% of batches
            log_interval = min(accumulation_steps, max(1, total_batches // 20))
        
        # Log initial message
        if epoch_num is not None:
            self.logger.info(f"Epoch {epoch_num + 1}/{self.num_epochs} | Starting training ({total_batches} batches)...")
        
        optimizer.zero_grad()
        
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
                    outputs = model(images, flips)

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
                            if self.confusion_targets is None:
                                raise RuntimeError(
                                    "flip_mode='inverted' requires confusion_targets "
                                    "to be loaded from baseline matrices."
                                )
                            losses.append(
                                flip_inverted_loss(
                                    outputs[flip_mask],
                                    labels[flip_mask],
                                    self.confusion_targets,
                                )
                            )
                    
                    loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=self.device)
            
            # Scale loss by accumulation steps for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass (accumulates gradients)
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            optimizer_step_occurred = False
            if (batch_idx + 1) % accumulation_steps == 0:
                # Apply gradient clipping if specified (before optimizer step)
                gradient_clip = exp_config.get('gradient_clip', None)
                if gradient_clip is not None:
                    if self.use_amp:
                        # Unscale gradients before clipping (for AMP)
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step()
                else:
                    if self.use_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()
                optimizer_step_occurred = True
            
            # Log progress periodically (on optimizer steps for gradient accumulation, or every log_interval)
            should_log = (
                optimizer_step_occurred and (batch_idx + 1) % log_interval == 0
            ) or (
                (batch_idx + 1) % log_interval == 0
            ) or (
                (batch_idx + 1) == total_batches
            ) or (
                accumulation_steps > 1 and (batch_idx + 1) == accumulation_steps  # Log first optimizer step
            )
            
            if should_log:
                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                batches_per_sec = batches_done / elapsed if elapsed > 0 else 0
                remaining_batches = total_batches - batches_done
                eta_seconds = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                
                current_loss = (total_loss / batches_done).item() if batches_done > 0 else 0.0
                current_acc = 100 * correct / total if total > 0 else 0.0
                
                epoch_str = f"Epoch {epoch_num + 1}/{self.num_epochs}" if epoch_num is not None else "Epoch"
                step_info = f"[Step {batches_done // accumulation_steps}]" if accumulation_steps > 1 and optimizer_step_occurred else ""
                self.logger.info(
                    f"{epoch_str} {step_info} | Batch {batches_done}/{total_batches} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.2f}% | "
                    f"Speed: {batches_per_sec:.1f} batch/s | ETA: {eta_min}m{eta_sec}s"
                )
            
            # Unscale loss for logging (multiply by accumulation_steps)
            total_loss += loss * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Handle remaining gradients if batch doesn't divide evenly
        if (batch_idx + 1) % accumulation_steps != 0:
            gradient_clip = exp_config.get('gradient_clip', None)
            self._optimizer_step(optimizer, model, gradient_clip)
        
        # Only call .item() once at the end
        avg_loss = (total_loss / len(train_loader)).item()
        accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch {epoch_num + 1 if epoch_num is not None else '?'} completed in {epoch_time:.1f}s | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
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
        
        # Hyperparameters are already set in __init__ via run_experiments.py
        # Use instance variables (already set correctly)
        paper_batch_size = self.batch_size
        paper_lr = self.learning_rate
        effective_batch_size = self.effective_batch_size
        
        # Add batch size info to exp_config for train_epoch
        exp_config['batch_size'] = paper_batch_size
        exp_config['effective_batch_size'] = effective_batch_size
        exp_config['gradient_clip'] = self.gradient_clip

        # If using inverted flip_mode, preload per-architecture confusion targets
        # built from baseline correlation matrices (top‑k confused classes).
        if exp_config.get('flip_mode') == 'inverted' and exp_config.get('fusion_type') != 'baseline':
            try:
                confusion_np = _load_confusion_topk_targets_for_arch(
                    exp_config['architecture'], k=5
                )
                self.confusion_targets = torch.tensor(
                    confusion_np, dtype=torch.float32, device=self.device
                )
                if self._is_main_process():
                    self.logger.info(
                        f"Loaded inverted confusion targets (top‑5) for architecture "
                        f"{exp_config['architecture']} from baseline matrices."
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to load confusion targets for inverted flip_mode: {e}",
                    exc_info=True,
                )
                return None
        else:
            self.confusion_targets = None
        
        # Check for existing checkpoint to resume WandB run
        checkpoints_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        latest_checkpoint = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        csv_log_path = os.path.join(checkpoints_dir, "training_log.csv")
        wandb_run_id = None
        
        # Initialize CSV log file (only on main process)
        if self._is_main_process():
            csv_exists = os.path.exists(csv_log_path)
            with open(csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not csv_exists:
                    # Write header
                    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                                   'test_loss', 'test_acc', 'learning_rate', 'best_val_acc', 'best_epoch'])
        
        if os.path.exists(latest_checkpoint):
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                wandb_run_id = checkpoint.get('wandb_run_id', None)
                if wandb_run_id:
                    self.logger.info(f"Found checkpoint with WandB run ID: {wandb_run_id}")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint for WandB resume: {e}")
        
        # Initialize wandb run with paper-matching hyperparameters
        wandb_config = self._get_wandb_config(exp_config, paper_batch_size, paper_lr, effective_batch_size)
        if wandb_run_id:
            wandb.init(
                project="flip-data-tinyimagenet",
                name=exp_name,
                id=wandb_run_id,
                resume="allow",
                config=wandb_config
            )
            self.logger.info(f"Resuming WandB run: {wandb_run_id}")
        else:
            wandb.init(
                project="flip-data-tinyimagenet",
                name=exp_name,
                config=wandb_config
            )
        
        # Create data loaders
        self.logger.info("Creating data loaders...")
        try:
            train_loader, val_loader, test_loader = create_dataset(
                dataset_name=exp_config['dataset'],
                batch_size=paper_batch_size,  # Use paper-matching batch size
                use_augmentation=exp_config['use_augmentation'],
                use_flip=(exp_config['fusion_type'] != 'baseline'),
                use_ddp=self.use_ddp,
                rank=self.rank,
                world_size=self.world_size
            )
            if self._is_main_process():
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
            
            use_compile = hasattr(torch, 'compile')
            if use_compile:
                # Skip compile for ViT only on single GPU (memory constraint)
                # With multiple GPUs (DDP), we have more memory per GPU
                skip_compile = (exp_config['architecture'].lower() == 'vit' and 
                               not self.use_ddp and torch.cuda.device_count() == 1)
                
                if not skip_compile:
                    try:
                        model = torch.compile(model, mode='reduce-overhead')
                        if self._is_main_process():
                            self.logger.info("✓ Model compiled with torch.compile (PyTorch 2.0+)")
                    except Exception as compile_error:
                        if self._is_main_process():
                            self.logger.warning(f"Model compilation failed, using standard model: {compile_error}")
                else:
                    if self._is_main_process():
                        self.logger.info("✓ Model created (torch.compile skipped for ViT on single GPU to save memory)")
            else:
                if self._is_main_process():
                    self.logger.info("✓ Model created (torch.compile not available)")
            
            # Wrap model with DDP if using distributed training
            if self.use_ddp:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.rank],
                    output_device=self.rank,
                    find_unused_parameters=False  # Set to True if model has unused parameters
                )
                if self.rank == 0:
                    self.logger.info("✓ Model wrapped with DistributedDataParallel")
            
            if self._is_main_process():
                self.logger.info(f"✓ Model created: {model_name} ({num_params:,} parameters)")
        except Exception as e:
            self.logger.error(f"✗ Failed to create model: {e}", exc_info=True)
            return None
        
        # Create optimizer using instance variables
        # Need to construct hyperparams dict for get_optimizer (it expects a dict)
        hyperparams_dict = {
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        optimizer = get_optimizer(model, hyperparams_dict)

        # Create scheduler using instance variables
        scheduler_name = self.scheduler
        scheduler_params = self.scheduler_params

        if scheduler_name == 'step':
            milestones = scheduler_params.get('milestones', [])
            gamma = scheduler_params.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosine_warmup':
            # Paper: Cosine decay with linear warmup (10k steps ≈ 3.4 epochs for ImageNet with batch 4096)
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
        elif scheduler_name == 'plateau':
            # Paper: VGG (Simonyan & Zisserman 2015) - Adaptive LR reduction
            mode = scheduler_params.get('mode', 'max')
            factor = scheduler_params.get('factor', 0.1)
            patience = scheduler_params.get('patience', 10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        else:
            scheduler = None
        # Log hyperparameters being used (from instance variables)
        optimizer_name = self.optimizer
        scheduler_desc = f"{scheduler_name}"
        if scheduler_name == 'step':
            milestones = scheduler_params.get('milestones', [])
            scheduler_desc = f"MultiStepLR (milestones={milestones}, gamma={scheduler_params.get('gamma', 0.1)})"
        elif scheduler_name == 'cosine_warmup':
            warmup = scheduler_params.get('warmup_epochs', 10)
            scheduler_desc = f"CosineWarmup (warmup={warmup} epochs)"
        elif scheduler_name == 'rmsprop_decay':
            scheduler_desc = "ExponentialLR (gamma=0.97)"
        elif scheduler_name == 'plateau':
            mode = scheduler_params.get('mode', 'max')
            factor = scheduler_params.get('factor', 0.1)
            patience = scheduler_params.get('patience', 10)
            scheduler_desc = f"ReduceLROnPlateau (mode={mode}, factor={factor}, patience={patience})"
        elif scheduler_name is None:
            scheduler_desc = "None (constant LR)"
        
        self.logger.info(f"Training hyperparameters (paper-matching):")
        self.logger.info(f"  Architecture: {exp_config['architecture']}")
        if effective_batch_size > paper_batch_size:
            accumulation_steps = effective_batch_size // paper_batch_size
            self.logger.info(f"  Batch size: {paper_batch_size} (actual) → {effective_batch_size} (effective, via {accumulation_steps}x gradient accumulation)")
        else:
            self.logger.info(f"  Batch size: {paper_batch_size} (from paper config)")
        self.logger.info(f"  Optimizer: {optimizer_name.upper()}")
        self.logger.info(f"  Learning rate: {paper_lr:.6f} (initial, from paper)")
        self.logger.info(f"  Weight decay: {self.weight_decay:.0e}")
        if self.gradient_clip is not None:
            self.logger.info(f"  Gradient clipping: {self.gradient_clip} (global norm)")
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
                    # Step scheduler to align with resumed epoch
                    # LambdaLR recalculates based on epoch number, so this is mainly for MultiStepLR/ExponentialLR
                    # ReduceLROnPlateau doesn't need stepping here as it's based on validation metrics
                    if scheduler_name != 'plateau':
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
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting training from epoch {start_epoch + 1} to {self.num_epochs}")
        self.logger.info(f"Total batches per epoch: {len(train_loader)}")
        if effective_batch_size > paper_batch_size:
            accumulation_steps = effective_batch_size // paper_batch_size
            self.logger.info(f"Gradient accumulation: {accumulation_steps} steps (optimizer updates every {accumulation_steps} batches)")
            self.logger.info(f"First optimizer update will occur after {accumulation_steps} batches")
        self.logger.info(f"{'='*80}\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, exp_config, epoch_num=epoch)

            # Step schedulers that don't require validation metrics (before evaluation)
            if scheduler is not None and scheduler_name != 'plateau':
                scheduler.step()
                # Some schedulers (LambdaLR/MultiStepLR) use optimizer.param_groups directly,
                # but get_last_lr() works for all standard schedulers.
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except AttributeError:
                    # ReduceLROnPlateau doesn't have get_last_lr()
                    current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate every N epochs, or on first/last epoch
            # In DDP mode, only evaluate on main process to avoid duplicate evaluation
            should_evaluate = (epoch == 0) or (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1 == self.num_epochs)
            evaluate_on_this_rank = should_evaluate and self._is_main_process()
            
            if evaluate_on_this_rank:
                # Unwrap DDP model for evaluation if needed
                eval_model = model.module if self.use_ddp else model
                val_loss, val_acc = self.evaluate(eval_model, val_loader, exp_config)
                test_loss, test_acc = self.evaluate(eval_model, test_loader, exp_config)
                
                # Update history
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                
                # Step ReduceLROnPlateau scheduler with validation metric (after evaluation)
                if scheduler is not None and scheduler_name == 'plateau':
                    scheduler.step(val_acc)  # Step with validation accuracy
                    current_lr = optimizer.param_groups[0]['lr']
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
            
            # Log to CSV file (only on main process)
            if self._is_main_process():
                with open(csv_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if evaluate_on_this_rank:
                        writer.writerow([
                            epoch + 1, train_loss, train_acc,
                            val_loss, val_acc, test_loss, test_acc,
                            current_lr, best_val_acc, best_epoch
                        ])
                    else:
                        # Use last known values for val/test if not evaluated this epoch
                        last_val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else ''
                        last_val_acc = history['val_acc'][-1] if len(history['val_acc']) > 0 else ''
                        last_test_loss = history['test_loss'][-1] if len(history['test_loss']) > 0 else ''
                        last_test_acc = history['test_acc'][-1] if len(history['test_acc']) > 0 else ''
                        writer.writerow([
                            epoch + 1, train_loss, train_acc,
                            last_val_loss, last_val_acc, last_test_loss, last_test_acc,
                            current_lr, best_val_acc, best_epoch
                        ])
            
            # Log to wandb (only log from main process, only log evaluation metrics when we actually evaluate)
            if self._is_main_process():
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'learning_rate': current_lr,
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                }
                if evaluate_on_this_rank:
                    log_dict.update({
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                    })
                wandb.log(log_dict)
            
            # Track best and early stopping (only when we actually evaluate, only on main process)
            if evaluate_on_this_rank:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                    last_improvement_epoch = epoch + 1  # Update actual epoch of last improvement
                    # Save best model locally (overwrite previous best model)
                    model_to_save = model.module if self.use_ddp else model
                    model_path = os.path.join(checkpoints_dir, "best_model.pth")
                    torch.save(model_to_save.state_dict(), model_path)
                    wandb.save(model_path)
            
            # Save latest checkpoint (for resume capability) - always save on main process
            if self._is_main_process():
                checkpoint = self._create_checkpoint_dict(
                    model, optimizer, scheduler, epoch, best_val_acc,
                    best_test_acc, best_epoch, last_improvement_epoch, history, start_time
                )
                
                # Save latest checkpoint (for immediate resume)
                latest_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
                try:
                    torch.save(checkpoint, latest_path)
                    #wandb.save(latest_path)
                except Exception as e:
                    self.logger.error(f"Failed to save latest checkpoint: {e}")
            
            # Log progress (every N epochs or when we evaluate, only on main process)
            if self._is_main_process() and ((epoch + 1) % LOG_INTERVAL_EPOCHS == 0 or epoch == 0 or evaluate_on_this_rank):
                if evaluate_on_this_rank:
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
                if self._is_main_process():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {epochs_since_improvement} epochs)")
                    wandb.log({'early_stopped': True, 'stopped_at_epoch': epoch + 1})
                break
        
        # Ensure we evaluate on the final epoch if training completed normally (only on main process)
        if self._is_main_process() and (len(history['val_loss']) == 0 or (len(history['train_loss']) > 0 and len(history['val_loss']) < len(history['train_loss']))):
            # Final epoch wasn't evaluated, evaluate now
            self.logger.info("Evaluating final model...")
            eval_model = model.module if self.use_ddp else model
            final_val_loss, final_val_acc = self.evaluate(eval_model, val_loader, exp_config)
            final_test_loss, final_test_acc = self.evaluate(eval_model, test_loader, exp_config)
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
        
        # Log final metrics to wandb (only on main process)
        if self._is_main_process():
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
        
        # Save final checkpoint (only on main process)
        if self._is_main_process():
            final_checkpoint = self._create_checkpoint_dict(
                model, optimizer, scheduler, actual_epochs - 1, best_val_acc,
                best_test_acc, best_epoch, last_improvement_epoch, history, start_time
            )
            final_checkpoint_path = os.path.join(checkpoints_dir, "final_checkpoint.pth")
            torch.save(final_checkpoint, final_checkpoint_path)
            wandb.save(final_checkpoint_path)
            
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



