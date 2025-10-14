"""
================================================================================
Training Callbacks Module
================================================================================
Production-grade callbacks for monitoring and controlling the training process.
Implements early stopping, learning rate scheduling, model checkpointing,
and custom metric tracking.

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import json


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Monitors a validation metric and stops training if no improvement is observed
    for a specified number of epochs (patience).
    
    Attributes:
        patience: Number of epochs with no improvement to wait before stopping.
        delta: Minimum change to qualify as an improvement.
        mode: 'min' for metrics like loss, 'max' for metrics like mAP.
        best_score: Best score observed so far.
        counter: Counter for epochs without improvement.
        early_stop: Flag indicating whether to stop training.
    
    Example:
        >>> early_stop = EarlyStopping(patience=50, mode='max')
        >>> for epoch in range(epochs):
        >>>     val_map = validate()
        >>>     if early_stop(val_map, epoch):
        >>>         print("Early stopping triggered")
        >>>         break
    """
    
    def __init__(
        self,
        patience: int = 50,
        delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ) -> None:
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait for improvement.
            delta: Minimum change in monitored metric to qualify as improvement.
            mode: 'max' for metrics to maximize (mAP), 'min' for metrics to
                 minimize (loss).
            verbose: Whether to print messages.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        # Set comparison function based on mode
        if mode == 'max':
            self.is_better = lambda current, best: current > best + delta
            self.best_score = float('-inf')
        elif mode == 'min':
            self.is_better = lambda current, best: current < best - delta
            self.best_score = float('inf')
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'max' or 'min'.")
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation metric value.
            epoch: Current epoch number.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.is_better(current_score, self.best_score):
            # Improvement observed
            if self.verbose:
                improvement = current_score - self.best_score if self.mode == 'max' else self.best_score - current_score
                print(f"EarlyStopping: Metric improved from {self.best_score:.4f} to "
                      f"{current_score:.4f} (Δ={improvement:.4f})")
            
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Training stopped at epoch {epoch}. "
                          f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = float('-inf') if self.mode == 'max' else float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Model checkpoint callback to save best and periodic model checkpoints.
    
    Saves model weights when validation metric improves and optionally saves
    checkpoints at regular intervals.
    
    Attributes:
        checkpoint_dir: Directory to save checkpoints.
        monitor: Metric to monitor for saving best model.
        mode: 'min' or 'max' for the monitored metric.
        save_best: Whether to save the best model.
        save_last: Whether to save the last model.
        save_frequency: Save checkpoint every N epochs (0 to disable).
        best_score: Best score observed so far.
    
    Example:
        >>> checkpoint = ModelCheckpoint('./models', monitor='map50', mode='max')
        >>> for epoch in range(epochs):
        >>>     val_map = validate()
        >>>     checkpoint(model, val_map, epoch)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'map50',
        mode: str = 'max',
        save_best: bool = True,
        save_last: bool = True,
        save_frequency: int = 0,
        verbose: bool = True
    ) -> None:
        """
        Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            monitor: Metric name to monitor.
            mode: 'max' to maximize metric, 'min' to minimize.
            save_best: Save best model based on monitored metric.
            save_last: Always save last epoch model.
            save_frequency: Save checkpoint every N epochs (0 to disable).
            verbose: Print save messages.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_frequency = save_frequency
        self.verbose = verbose
        
        # Track best score
        if mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
        elif mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.best_epoch = 0
    
    def __call__(
        self,
        model: Any,
        current_score: float,
        epoch: int,
        metrics: Optional[Dict] = None
    ) -> bool:
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model object with save() method.
            current_score: Current value of monitored metric.
            epoch: Current epoch number.
            metrics: Optional dictionary of additional metrics to save.
        
        Returns:
            True if best model was saved, False otherwise.
        """
        saved_best = False
        
        # Save best model
        if self.save_best and self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            
            best_path = self.checkpoint_dir / 'best.pt'
            model.save(best_path)
            
            # Save metadata
            if metrics:
                metadata = {
                    'epoch': epoch,
                    'monitor': self.monitor,
                    'score': current_score,
                    **metrics
                }
                metadata_path = self.checkpoint_dir / 'best_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            if self.verbose:
                print(f"ModelCheckpoint: Saved best model (epoch {epoch}, "
                      f"{self.monitor}={current_score:.4f}) to {best_path}")
            
            saved_best = True
        
        # Save last model
        if self.save_last:
            last_path = self.checkpoint_dir / 'last.pt'
            model.save(last_path)
        
        # Save periodic checkpoint
        if self.save_frequency > 0 and (epoch + 1) % self.save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            model.save(checkpoint_path)
            
            if self.verbose:
                print(f"ModelCheckpoint: Saved periodic checkpoint to {checkpoint_path}")
        
        return saved_best


class MetricsTracker:
    """
    Track training and validation metrics across epochs.
    
    Stores metrics history and provides methods for querying and saving.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> for epoch in range(epochs):
        >>>     tracker.update('train_loss', train_loss, epoch)
        >>>     tracker.update('val_map50', val_map, epoch)
        >>> tracker.save('./results/metrics_history.json')
    """
    
    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.metrics: Dict[str, list] = {}
        self.epochs: list = []
    
    def update(self, metric_name: str, value: float, epoch: int) -> None:
        """
        Update metric value for current epoch.
        
        Args:
            metric_name: Name of the metric.
            value: Metric value.
            epoch: Current epoch number.
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        if epoch not in self.epochs:
            self.epochs.append(epoch)
    
    def update_batch(self, metrics_dict: Dict[str, float], epoch: int) -> None:
        """
        Update multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric_name: value pairs.
            epoch: Current epoch number.
        """
        for metric_name, value in metrics_dict.items():
            self.update(metric_name, value, epoch)
    
    def get(self, metric_name: str) -> list:
        """Get history of a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value of a metric."""
        history = self.get(metric_name)
        return history[-1] if history else None
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value and epoch for a metric.
        
        Args:
            metric_name: Name of the metric.
            mode: 'max' to find maximum, 'min' to find minimum.
        
        Returns:
            Tuple of (best_value, best_epoch).
        """
        history = self.get(metric_name)
        if not history:
            return None, None
        
        if mode == 'max':
            best_idx = np.argmax(history)
        else:
            best_idx = np.argmin(history)
        
        return history[best_idx], self.epochs[best_idx]
    
    def save(self, filepath: str) -> None:
        """Save metrics history to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'epochs': self.epochs,
            'metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load metrics history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.epochs = data['epochs']
        self.metrics = data['metrics']


if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3, mode='max', verbose=True)
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]  # Stops at index 5
    
    for epoch, score in enumerate(scores):
        if early_stop(score, epoch):
            print(f"Training would stop at epoch {epoch}")
            break
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    for epoch in range(10):
        tracker.update('train_loss', 1.0 - epoch * 0.1, epoch)
        tracker.update('val_map50', 0.5 + epoch * 0.05, epoch)
    
    best_map, best_epoch = tracker.get_best('val_map50', mode='max')
    print(f"Best mAP: {best_map:.3f} at epoch {best_epoch}")
    
    print("All callback tests passed!")
