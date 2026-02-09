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
from losses import normal_loss, flip_all_loss, flip_inverted_loss


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
        total_loss = 0.0
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
                            losses.append(flip_all_loss(outputs[flip_mask], labels[flip_mask]))
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
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
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
        
        # Initialize wandb run
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
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'early_stop_patience': self.early_stop_patience,
                'seed': 42,
            }
        )
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"EXPERIMENT: {exp_name}")
        self.logger.info("="*80)
        
        # Validate experiment
        if not validate_experiment(exp_config):
            self.logger.error(f"Invalid experiment config: {exp_config}")
            return None
        
        # Create data loaders
        self.logger.info("Creating data loaders...")
        try:
            train_loader, val_loader, test_loader = create_dataset(
                dataset_name=exp_config['dataset'],
                batch_size=self.batch_size,
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
            self.logger.info(f"✓ Model created: {model_name} ({num_params:,} parameters)")
        except Exception as e:
            self.logger.error(f"✗ Failed to create model: {e}", exc_info=True)
            return None
        
        # Get hyperparameters for weight decay
        from hyperparameters import get_hyperparameters
        hyperparams = get_hyperparameters(exp_config['architecture'])
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        
        # Create optimizer with weight decay
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        
        # Create learning rate scheduler (cosine annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        
        # Log hyperparameters being used
        self.logger.info(f"Training hyperparameters:")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Learning rate: {self.learning_rate:.6f} (initial)")
        self.logger.info(f"  Weight decay: {weight_decay:.0e}")
        self.logger.info(f"  LR Scheduler: CosineAnnealingLR (T_max={self.num_epochs}, eta_min=1e-6)")
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': []
        }
        
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        start_time = datetime.now()
        
        # Training loop with early stopping
        for epoch in range(self.num_epochs):
            # Train (optimizer.step() is called inside train_epoch)
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, exp_config)
            
            # Update learning rate AFTER optimizer.step() (PyTorch best practice)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate
            val_loss, val_acc = self.evaluate(model, val_loader, exp_config)
            test_loss, test_acc = self.evaluate(model, test_loader, exp_config)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'learning_rate': current_lr,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
            })
            
            # Track best and early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model to wandb
                model_path = f"best_model_epoch_{best_epoch}.pth"
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
                os.remove(model_path)  # Remove local file after uploading
            else:
                epochs_without_improvement += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                    f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
                    f"Test: {test_loss:.4f}/{test_acc:.2f}% | "
                    f"Best: {best_val_acc:.2f}% (epoch {best_epoch}) | "
                    f"LR: {current_lr:.6f} | "
                    f"No improvement: {epochs_without_improvement}/{self.early_stop_patience}"
                )
            
            # Early stopping
            if epochs_without_improvement >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {self.early_stop_patience} epochs)")
                wandb.log({'early_stopped': True, 'stopped_at_epoch': epoch + 1})
                break
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        actual_epochs = len(history['train_loss'])
        
        # Log final metrics to wandb
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
            'best_epoch': best_epoch,
            'total_training_time_seconds': training_time,
            'actual_epochs': actual_epochs,
            'num_parameters': num_params,
        })
        
        # Summary metrics for wandb
        wandb.summary.update({
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
            'best_epoch': best_epoch,
            'total_training_time_seconds': training_time,
            'actual_epochs': actual_epochs,
            'num_parameters': num_params,
        })
        
        self.logger.info(f"✓ Training completed: Best val acc: {best_val_acc:.2f}%, Best test acc: {best_test_acc:.2f}%")
        self.logger.info(f"  Trained for {actual_epochs} epochs (out of {self.num_epochs} max)")
        self.logger.info(f"  Training time: {training_time/3600:.2f} hours")
        
        # Finish wandb run
        wandb.finish()
        
        return {
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
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



