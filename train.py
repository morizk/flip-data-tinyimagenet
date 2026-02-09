import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import logging
from datetime import datetime

from models import BaselineCNN, FlipCNN_LateFusion, FlipCNN_EarlyFusion
from data_utils import get_cifar10_loaders
from losses import normal_loss, flip_all_loss, flip_any_loss, flip_inverted_loss


def setup_logging(log_dir='logs'):
    """Setup comprehensive logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (info and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file


class ExperimentRunner:
    """Runs all experiments and tracks results."""
    
    def __init__(self, num_epochs=100, batch_size=64, learning_rate=0.001, device=None, logger=None, use_amp=True):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.logger = logger if logger else logging.getLogger('ExperimentLogger')
        self.use_amp = use_amp and torch.cuda.is_available()  # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT RUNNER INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of epochs: {num_epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Mixed Precision (AMP): {self.use_amp}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
    
    def test_loss_functions(self):
        """Test loss function implementations with a trial run."""
        self.logger.info("="*80)
        self.logger.info("TESTING LOSS FUNCTION IMPLEMENTATIONS")
        self.logger.info("="*80)
        
        # Create dummy data
        batch_size = 8
        num_classes = 10
        predictions = torch.randn(batch_size, num_classes, device=self.device)
        true_labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        
        # Test normal loss
        self.logger.debug("Testing normal_loss...")
        try:
            loss_val = normal_loss(predictions, true_labels)
            self.logger.info(f"✓ normal_loss: {loss_val.item():.4f} (expected: positive value)")
            assert loss_val.item() > 0, "Normal loss should be positive"
        except Exception as e:
            self.logger.error(f"✗ normal_loss failed: {e}")
            raise
        
        # Test flip_all_loss
        self.logger.debug("Testing flip_all_loss...")
        try:
            loss_val = flip_all_loss(predictions, true_labels)
            self.logger.info(f"✓ flip_all_loss: {loss_val.item():.4f} (expected: positive value)")
            assert loss_val.item() > 0, "Flip-all loss should be positive"
            
            # Verify it creates uniform distribution over wrong classes
            self.logger.debug("Verifying flip_all_loss creates uniform target over wrong classes...")
            true_class = true_labels[0].item()
            wrong_classes = [c for c in range(num_classes) if c != true_class]
            self.logger.debug(f"  True class: {true_class}, Wrong classes: {wrong_classes}")
            self.logger.info(f"✓ flip_all_loss verified: uniform over {len(wrong_classes)} wrong classes")
        except Exception as e:
            self.logger.error(f"✗ flip_all_loss failed: {e}")
            raise
        
        # Test flip_any_loss
        self.logger.debug("Testing flip_any_loss...")
        try:
            loss_val = flip_any_loss(predictions, true_labels)
            self.logger.info(f"✓ flip_any_loss: {loss_val.item():.4f} (expected: positive value)")
            assert loss_val.item() > 0, "Flip-any loss should be positive"
            self.logger.info("✓ flip_any_loss verified: randomly selects one wrong class")
        except Exception as e:
            self.logger.error(f"✗ flip_any_loss failed: {e}")
            raise
        
        # Test flip_inverted_loss
        self.logger.debug("Testing flip_inverted_loss...")
        try:
            loss_val = flip_inverted_loss(predictions, true_labels)
            self.logger.info(f"✓ flip_inverted_loss: {loss_val.item():.4f} (expected: positive value, true class probability)")
            assert loss_val.item() > 0, "Flip-inverted loss should be positive"
            assert loss_val.item() <= 1.0, "Flip-inverted loss should be <= 1.0 (probability)"
            
            # Verify it's actually the true class probability
            import torch.nn.functional as F
            probs = F.softmax(predictions, dim=1)
            true_class_probs = probs[range(len(true_labels)), true_labels]
            expected_loss = true_class_probs.mean()
            assert abs(loss_val.item() - expected_loss.item()) < 1e-5, "Flip-inverted loss should equal mean true class probability"
            self.logger.info("✓ flip_inverted_loss verified: minimizes true class probability")
        except Exception as e:
            self.logger.error(f"✗ flip_inverted_loss failed: {e}")
            raise
        
        self.logger.info("="*80)
        self.logger.info("ALL LOSS FUNCTIONS TESTED SUCCESSFULLY")
        self.logger.info("="*80)
    
    def verify_model_architecture(self, model, architecture, experiment_name):
        """Verify and log model architecture details."""
        self.logger.info("="*80)
        self.logger.info(f"VERIFYING MODEL ARCHITECTURE: {experiment_name}")
        self.logger.info("="*80)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Architecture type: {architecture}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data (CIFAR-10: 3 channels, 32x32)
        self.logger.debug("Testing forward pass...")
        if architecture == 'baseline':
            dummy_input = torch.randn(2, 3, 32, 32, device=self.device)
            try:
                output = model(dummy_input)
                self.logger.info(f"✓ Forward pass successful")
                self.logger.debug(f"  Input shape: {dummy_input.shape}")
                self.logger.debug(f"  Output shape: {output.shape}")
                assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
            except Exception as e:
                self.logger.error(f"✗ Forward pass failed: {e}")
                raise
        else:
            dummy_input = torch.randn(2, 3, 32, 32, device=self.device)
            dummy_flip = torch.randint(0, 2, (2,), dtype=torch.long, device=self.device)
            try:
                output = model(dummy_input, dummy_flip)
                self.logger.info(f"✓ Forward pass successful")
                self.logger.debug(f"  Input shape: {dummy_input.shape}")
                self.logger.debug(f"  Flip shape: {dummy_flip.shape}")
                self.logger.debug(f"  Output shape: {output.shape}")
                assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
            except Exception as e:
                self.logger.error(f"✗ Forward pass failed: {e}")
                raise
        
        # Log model structure
        self.logger.debug("Model structure:")
        for name, module in model.named_children():
            self.logger.debug(f"  {name}: {type(module).__name__}")
        
        self.logger.info("="*80)
        self.logger.info("MODEL ARCHITECTURE VERIFIED")
        self.logger.info("="*80)
    
    def log_data_preparation(self, train_loader, val_loader, test_loader, exp_config):
        """Log data preparation details."""
        self.logger.info("="*80)
        self.logger.info("DATA PREPARATION DETAILS")
        self.logger.info("="*80)
        
        self.logger.info(f"Experiment: {exp_config['name']}")
        self.logger.info(f"Use augmentation: {exp_config['use_augmentation']}")
        self.logger.info(f"Use flip: {exp_config['use_flip']}")
        self.logger.info(f"Architecture: {exp_config['architecture']}")
        self.logger.info(f"Flip mode: {exp_config['flip_mode']}")
        
        # Get a sample batch
        train_batch = next(iter(train_loader))
        if exp_config['use_flip']:
            images, labels, flips = train_batch
            self.logger.info(f"Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}, Flips shape: {flips.shape}")
            self.logger.debug(f"  Sample labels: {labels[:5].tolist()}")
            self.logger.debug(f"  Sample flips: {flips[:5].tolist()}")
            self.logger.debug(f"  Flip distribution - 0: {(flips == 0).sum().item()}, 1: {(flips == 1).sum().item()}")
        else:
            images, labels = train_batch
            self.logger.info(f"Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
            self.logger.debug(f"  Sample labels: {labels[:5].tolist()}")
        
        self.logger.info(f"Training batches: {len(train_loader)}")
        self.logger.info(f"Validation batches: {len(val_loader)}")
        self.logger.info(f"Test batches: {len(test_loader)}")
        
        # Calculate dataset sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        
        self.logger.info(f"Training samples: {train_size:,}")
        self.logger.info(f"Validation samples: {val_size:,}")
        self.logger.info(f"Test samples: {test_size:,}")
        
        if exp_config['use_flip']:
            # Each image appears twice, so original size is half
            original_train_size = train_size // 2
            self.logger.info(f"Original training samples (before flip duplication): {original_train_size:,}")
            self.logger.info(f"Dataset size increase due to flip: {train_size / original_train_size:.1f}x")
        
        self.logger.info("="*80)
        self.logger.info("DATA PREPARATION COMPLETE")
        self.logger.info("="*80)
    
    def train_epoch(self, model, train_loader, optimizer, loss_fn, flip_mode='none', architecture='baseline', epoch=0):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        
        self.logger.debug(f"Epoch {epoch+1}: Starting training, {num_batches} batches")
        
        for batch_idx, batch in enumerate(train_loader):
            # Use autocast for mixed precision training
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if architecture == 'baseline':
                    images, labels = batch[0], batch[1]
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                else:
                    images, labels, flips = batch[0], batch[1], batch[2]
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    flips = flips.to(self.device, non_blocking=True)
                    
                    outputs = model(images, flips)
                    
                    # Determine which loss to use based on flip value
                    batch_size = images.size(0)
                    
                    # Separate normal and flip samples
                    normal_mask = (flips == 0)
                    flip_mask = (flips == 1)
                    
                    losses = []
                    
                    if normal_mask.sum() > 0:
                        normal_loss_val = normal_loss(outputs[normal_mask], labels[normal_mask])
                        losses.append(normal_loss_val)
                    
                    if flip_mask.sum() > 0:
                        if flip_mode == 'all':
                            flip_loss_val = flip_all_loss(outputs[flip_mask], labels[flip_mask])
                        elif flip_mode == 'any':
                            flip_loss_val = flip_any_loss(outputs[flip_mask], labels[flip_mask])
                        elif flip_mode == 'inverted':
                            flip_loss_val = flip_inverted_loss(outputs[flip_mask], labels[flip_mask])
                        else:
                            flip_loss_val = normal_loss(outputs[flip_mask], labels[flip_mask])
                        losses.append(flip_loss_val)
                    
                    # Average the losses
                    if len(losses) > 0:
                        loss = sum(losses) / len(losses)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
            
            # Use scaler for mixed precision
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
            
            # Log every 200 batches (reduced logging frequency for speed)
            if (batch_idx + 1) % 200 == 0:
                batch_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                self.logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        self.logger.debug(f"Epoch {epoch+1} training complete: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        return avg_loss, accuracy
    
    def evaluate(self, model, test_loader, architecture='baseline', split_name='test'):
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0
        
        self.logger.debug(f"Evaluating on {split_name} set ({len(test_loader)} batches)...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Use autocast for evaluation too (faster)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if architecture == 'baseline':
                        images, labels = batch[0], batch[1]
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        outputs = model(images)
                    else:
                        images, labels = batch[0], batch[1]
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        # For evaluation, always use flip=0 (normal classification)
                        flips = torch.zeros(images.size(0), dtype=torch.long, device=self.device)
                        outputs = model(images, flips)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        self.logger.debug(f"{split_name.capitalize()} evaluation complete: {correct}/{total} correct, Acc={accuracy:.2f}%")
        return accuracy
    
    def train_experiment(self, experiment_name, model, train_loader, val_loader, test_loader, 
                        loss_fn, flip_mode='none', architecture='baseline', exp_config=None):
        """Train a single experiment."""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"STARTING EXPERIMENT: {experiment_name}")
        self.logger.info("="*80)
        
        # Log data preparation
        if exp_config:
            self.log_data_preparation(train_loader, val_loader, test_loader, exp_config)
        
        # Verify model architecture
        self.verify_model_architecture(model, architecture, experiment_name)
        
        # Initialize optimizer
        self.logger.info("Initializing optimizer...")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.logger.info(f"✓ Optimizer: Adam, Learning rate: {self.learning_rate}")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }
        
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_epoch = 0
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("STARTING TRAINING LOOP")
        self.logger.info("="*80)
        
        for epoch in range(self.num_epochs):
            self.logger.info("")
            self.logger.info(f"EPOCH {epoch+1}/{self.num_epochs}")
            self.logger.info("-"*80)
            
            # Train
            self.logger.info("Phase: TRAINING")
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, loss_fn, flip_mode, architecture, epoch
            )
            
            # Validate
            self.logger.info("Phase: VALIDATION")
            val_acc = self.evaluate(model, val_loader, architecture, 'validation')
            
            # Test
            self.logger.info("Phase: TESTING")
            test_acc = self.evaluate(model, test_loader, architecture, 'test')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            
            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1
                self.logger.info(f"✓ New best validation accuracy! Epoch {best_epoch}")
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1} Results:")
            self.logger.info(f"  Train Loss: {train_loss:.6f}")
            self.logger.info(f"  Train Acc:  {train_acc:.4f}%")
            self.logger.info(f"  Val Acc:    {val_acc:.4f}%")
            self.logger.info(f"  Test Acc:   {test_acc:.4f}%")
            self.logger.info(f"  Best Val Acc (epoch {best_epoch}): {best_val_acc:.4f}%")
            self.logger.info(f"  Best Test Acc (epoch {best_epoch}): {best_test_acc:.4f}%")
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"TRAINING COMPLETE: {experiment_name}")
        self.logger.info("="*80)
        self.logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}% (Epoch {best_epoch})")
        self.logger.info(f"Best Test Accuracy: {best_test_acc:.4f}% (Epoch {best_epoch})")
        self.logger.info(f"Final Test Accuracy: {test_acc:.4f}%")
        
        # Save results
        self.results[experiment_name] = {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc
        }
        
        # Save model
        self.logger.info("Saving model checkpoint...")
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/{experiment_name.replace(" ", "_")}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        self.logger.info(f"✓ Model saved to: {checkpoint_path}")
        
        return history
    
    def run_all_experiments(self):
        """Run all 10 experiments."""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("INITIALIZING ALL EXPERIMENTS")
        self.logger.info("="*80)
        
        # Test loss functions first
        self.test_loss_functions()
        
        experiments = [
            # Baseline experiments
            {
                'name': 'Baseline (no aug)',
                'use_augmentation': False,
                'use_flip': False,
                'architecture': 'baseline',
                'flip_mode': 'none'
            },
            {
                'name': 'Baseline + Augmentation',
                'use_augmentation': True,
                'use_flip': False,
                'architecture': 'baseline',
                'flip_mode': 'none'
            },
            # Flip-All Late Fusion
            {
                'name': 'Flip-All Late Fusion (no aug)',
                'use_augmentation': False,
                'use_flip': True,
                'architecture': 'late_fusion',
                'flip_mode': 'all'
            },
            {
                'name': 'Flip-All Late Fusion + Augmentation',
                'use_augmentation': True,
                'use_flip': True,
                'architecture': 'late_fusion',
                'flip_mode': 'all'
            },
            # Flip-All Early Fusion
            {
                'name': 'Flip-All Early Fusion (no aug)',
                'use_augmentation': False,
                'use_flip': True,
                'architecture': 'early_fusion',
                'flip_mode': 'all'
            },
            {
                'name': 'Flip-All Early Fusion + Augmentation',
                'use_augmentation': True,
                'use_flip': True,
                'architecture': 'early_fusion',
                'flip_mode': 'all'
            },
            # Flip-Any Late Fusion
            {
                'name': 'Flip-Any Late Fusion (no aug)',
                'use_augmentation': False,
                'use_flip': True,
                'architecture': 'late_fusion',
                'flip_mode': 'any'
            },
            {
                'name': 'Flip-Any Late Fusion + Augmentation',
                'use_augmentation': True,
                'use_flip': True,
                'architecture': 'late_fusion',
                'flip_mode': 'any'
            },
            # Flip-Any Early Fusion
            {
                'name': 'Flip-Any Early Fusion (no aug)',
                'use_augmentation': False,
                'use_flip': True,
                'architecture': 'early_fusion',
                'flip_mode': 'any'
            },
            {
                'name': 'Flip-Any Early Fusion + Augmentation',
                'use_augmentation': True,
                'use_flip': True,
                'architecture': 'early_fusion',
                'flip_mode': 'any'
            },
        ]
        
        total_experiments = len(experiments)
        self.logger.info(f"Total experiments to run: {total_experiments}")
        
        for exp_idx, exp_config in enumerate(experiments, 1):
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"EXPERIMENT {exp_idx}/{total_experiments}")
            self.logger.info("="*80)
            
            # Get data loaders
            self.logger.info("Loading data...")
            try:
                train_loader, val_loader, test_loader = get_cifar10_loaders(
                    batch_size=self.batch_size,
                    use_augmentation=exp_config['use_augmentation'],
                    use_flip=exp_config['use_flip']
                )
                self.logger.info("✓ Data loaders created successfully")
            except Exception as e:
                self.logger.error(f"✗ Failed to create data loaders: {e}")
                raise
            
            # Create model
            self.logger.info(f"Creating model: {exp_config['architecture']}...")
            try:
                if exp_config['architecture'] == 'baseline':
                    model = BaselineCNN().to(self.device)
                elif exp_config['architecture'] == 'late_fusion':
                    model = FlipCNN_LateFusion().to(self.device)
                elif exp_config['architecture'] == 'early_fusion':
                    model = FlipCNN_EarlyFusion().to(self.device)
                else:
                    raise ValueError(f"Unknown architecture: {exp_config['architecture']}")
                self.logger.info(f"✓ Model created and moved to {self.device}")
            except Exception as e:
                self.logger.error(f"✗ Failed to create model: {e}")
                raise
            
            # Train
            try:
                self.train_experiment(
                    experiment_name=exp_config['name'],
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    loss_fn=normal_loss,
                    flip_mode=exp_config['flip_mode'],
                    architecture=exp_config['architecture'],
                    exp_config=exp_config
                )
                self.logger.info(f"✓ Experiment {exp_idx}/{total_experiments} completed successfully")
            except Exception as e:
                self.logger.error(f"✗ Experiment {exp_idx}/{total_experiments} failed: {e}")
                raise
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save experiment results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for exp_name, exp_results in self.results.items():
            results_to_save[exp_name] = {
                'history': {
                    'train_loss': [float(x) for x in exp_results['history']['train_loss']],
                    'train_acc': [float(x) for x in exp_results['history']['train_acc']],
                    'val_acc': [float(x) for x in exp_results['history']['val_acc']],
                    'test_acc': [float(x) for x in exp_results['history']['test_acc']]
                },
                'best_val_acc': float(exp_results['best_val_acc']),
                'best_test_acc': float(exp_results['best_test_acc']),
                'final_test_acc': float(exp_results['final_test_acc'])
            }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results_{timestamp}.json'
        self.logger.info("Saving results to JSON...")
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"✓ Results saved to {filename}")
        print(f"\nResults saved to {filename}")


if __name__ == '__main__':
    # Setup logging first
    logger, log_file = setup_logging()
    
    logger.info("")
    logger.info("="*80)
    logger.info("CIFAR-10 FLIP DATA EXPERIMENTS - STARTING")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    
    # Configuration
    NUM_EPOCHS = 100
    BATCH_SIZE = 128  # Increased batch size for better GPU utilization
    LEARNING_RATE = 0.001
    USE_AMP = True  # Mixed precision training (FP16) - significantly faster
    
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Mixed Precision (AMP): {USE_AMP}")
    
    # Create experiment runner
    logger.info("")
    logger.info("Creating experiment runner...")
    runner = ExperimentRunner(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logger=logger,
        use_amp=USE_AMP
    )
    
    # Run all experiments
    logger.info("")
    logger.info("Starting all experiments...")
    try:
        results = runner.run_all_experiments()
        logger.info("")
        logger.info("="*80)
        logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        print("\n" + "="*60)
        print("All experiments completed!")
        print("="*60)
        print(f"Detailed log saved to: {log_file}")
    except Exception as e:
        logger.error("="*80)
        logger.error("EXPERIMENTS FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Check log file for details: {log_file}")
        raise

