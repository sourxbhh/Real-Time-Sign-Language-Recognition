"""
Training Pipeline for Sign Language Recognition
================================================

This module provides a complete training pipeline with:
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training visualization
- Mixed precision support

Training Strategy:
------------------

1. **Learning Rate Schedule**
   - Start with warmup (gradually increase LR)
   - Use cosine annealing for smooth decay
   - This helps with convergence and final accuracy

2. **Class Imbalance Handling**
   - Weighted loss function (CrossEntropyLoss with weights)
   - Balanced sampling (WeightedRandomSampler)
   - Data augmentation for minority classes

3. **Regularization**
   - Dropout in model (already included)
   - Weight decay (L2 regularization)
   - Early stopping to prevent overfitting

4. **Evaluation Metrics**
   - Accuracy (overall and per-class)
   - Confusion matrix
   - F1 score (macro and weighted)
   - Per-class precision and recall

Best Practices:
---------------
1. Start with small model and increase complexity if needed
2. Monitor validation loss, not just accuracy
3. Use learning rate finder to pick good initial LR
4. Save checkpoints frequently
5. Log everything for reproducibility
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.recognition.model import SignLanguageModel, SignLanguageModelStatic
from training.dataset import ASLDataset, create_data_loaders

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training configuration.

    All hyperparameters in one place for easy experimentation.
    """
    # Data
    data_dir: str = "data/processed"
    sequence_length: int = 1  # 1 for static, >1 for temporal

    # Model
    model_type: str = "static"  # 'static' or 'temporal'
    num_classes: int = 29

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Learning rate schedule
    scheduler_type: str = "cosine"  # 'cosine', 'plateau', 'step'
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_every: int = 5

    # Hardware
    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True

    # Logging
    log_dir: str = "logs"
    log_every: int = 10


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if it doesn't improve
    for a specified number of epochs.

    Why Early Stopping?
    -------------------
    - Prevents overfitting by stopping before model memorizes training data
    - Saves computational resources
    - Often leads to better generalization

    How it Works:
    -------------
    1. Track best validation loss seen so far
    2. If current loss is worse than best + delta, increment counter
    3. If counter exceeds patience, stop training
    4. Save best model weights for restoration
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset counter
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_weights = None

    def __call__(
        self,
        score: float,
        model: nn.Module
    ) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric
            model: Model to save weights from

        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore model to best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class Trainer:
    """
    Training manager for sign language recognition models.

    This class handles the complete training loop including:
    - Model initialization
    - Optimizer and scheduler setup
    - Training and validation loops
    - Checkpointing
    - Logging and visualization

    Usage:
    ------
    >>> config = TrainingConfig(epochs=50, batch_size=32)
    >>> trainer = Trainer(config)
    >>> history = trainer.train()
    >>> trainer.save_model('models/final_model.pt')
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components (done in train())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        if self.config.model_type == 'static':
            model = SignLanguageModelStatic(num_classes=self.config.num_classes)
        else:
            model = SignLanguageModel(num_classes=self.config.num_classes)

        model = model.to(self.device)

        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {num_params:,} parameters")

        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999)
        )

        return optimizer

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.epochs // 3,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr
            )
        else:  # step
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )

        return scheduler

    def train(self) -> Dict[str, List[float]]:
        """
        Run the complete training loop.

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            num_workers=self.config.num_workers
        )

        # Initialize model, optimizer, etc.
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss function with class weights for imbalanced data
        train_dataset = train_loader.dataset
        class_weights = train_dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )

        # Training loop
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self._validate(val_loader)

            # Update scheduler
            if self.config.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"LR: {current_lr:.6f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")

            # Early stopping check
            if early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopping.restore_best_weights(self.model)
                break

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.2f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final model and history
        self._save_checkpoint("final_model.pt")
        self._save_history()

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1} [Train]")

        for batch_idx, (landmarks, labels) in enumerate(pbar):
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(landmarks)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(landmarks)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for landmarks, labels in tqdm(val_loader, desc=f"Epoch {self.epoch + 1} [Val]"):
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(landmarks)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set with detailed metrics.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for landmarks, labels in tqdm(test_loader, desc="Evaluating"):
                landmarks = landmarks.to(self.device)

                outputs = self.model(landmarks)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        accuracy = (predictions == labels).mean()

        # Per-class accuracy
        class_names = ASLDataset.CLASSES
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class_acc[class_name] = (predictions[mask] == labels[mask]).mean()

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)

        results = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")

        return results

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }

        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def _save_history(self) -> None:
        """Save training history."""
        path = Path(self.config.log_dir) / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is None:
            self.model = self._create_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def save_model(self, path: str) -> None:
        """Save model weights only (for deployment)."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved: {path}")


# =============================================================================
# Training Script Entry Point
# =============================================================================

def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model-type', type=str, default='static',
                       choices=['static', 'temporal'],
                       help='Model type')
    parser.add_argument('--sequence-length', type=int, default=1,
                       help='Sequence length for temporal model')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model_type,
        sequence_length=args.sequence_length
    )

    # Train
    trainer = Trainer(config)
    history = trainer.train()

    # Save final model
    trainer.save_model('models/sign_language_model.pt')

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
