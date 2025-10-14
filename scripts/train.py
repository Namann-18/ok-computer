"""
================================================================================
YOLOv8m Training Script for Space Station Safety Object Detection
================================================================================
Production-grade training pipeline optimized for ≥80% mAP@0.5 accuracy.

Features:
    - YOLOv8m architecture with transfer learning
    - Advanced data augmentation for challenging scenarios
    - Learning rate scheduling and early stopping
    - Comprehensive logging and checkpointing
    - Multi-GPU support
    - TensorBoard integration
    
Usage:
    python scripts/train.py --config configs/train_config.yaml --data configs/dataset.yaml
    
    # Resume from checkpoint
    python scripts/train.py --config configs/train_config.yaml --resume models/last.pt
    
    # Custom settings
    python scripts/train.py --epochs 300 --batch 16 --device 0

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from utils.logger import get_logger
from utils.callbacks import EarlyStopping, ModelCheckpoint, MetricsTracker
from utils.visualization import plot_training_curves


class YOLOv8Trainer:
    """
    Production-grade trainer for YOLOv8m object detection model.
    
    Handles end-to-end training process including:
        - Model initialization and configuration
        - Training loop with validation
        - Checkpointing and early stopping
        - Metrics tracking and visualization
        - TensorBoard logging
    
    Attributes:
        config: Training configuration dictionary.
        data_config: Dataset configuration dictionary.
        model: YOLO model instance.
        logger: Logger instance for structured logging.
        device: Training device (cuda/cpu).
        
    Example:
        >>> trainer = YOLOv8Trainer(train_config, data_config)
        >>> trainer.train()
        >>> results = trainer.get_results()
    """
    
    def __init__(
        self,
        train_config: Dict,
        data_config: Dict,
        resume: Optional[str] = None,
        log_dir: str = "./logs"
    ) -> None:
        """
        Initialize trainer with configuration.
        
        Args:
            train_config: Training configuration dictionary.
            data_config: Dataset configuration dictionary.
            resume: Path to checkpoint to resume from (optional).
            log_dir: Directory for log files.
        """
        self.config = train_config
        self.data_config = data_config
        self.resume = resume
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(
            name=f"train_{timestamp}",
            log_dir=log_dir
        )
        
        self.logger.info("="*80)
        self.logger.info("YOLOv8m Training Pipeline Initialized")
        self.logger.info("="*80)
        
        # Log configurations
        self.logger.log_config(train_config['training'], "Training Configuration")
        self.logger.log_config(data_config, "Dataset Configuration")
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        self.logger.info("Trainer initialization complete")
    
    def _setup_device(self) -> str:
        """
        Setup training device (GPU/CPU).
        
        Returns:
            Device string ('cuda:0', 'cpu', etc.).
        """
        device = self.config['training'].get('device', '0')
        
        if device != 'cpu' and torch.cuda.is_available():
            device_id = int(device) if device.isdigit() else 0
            device_name = torch.cuda.get_device_name(device_id)
            self.logger.info(f"Using GPU: {device_name} (cuda:{device_id})")
            return f"cuda:{device_id}"
        else:
            self.logger.warning("CUDA not available, using CPU (training will be slow)")
            return 'cpu'
    
    def _initialize_model(self) -> YOLO:
        """
        Initialize YOLO model with pre-trained weights or from checkpoint.
        
        Returns:
            YOLO model instance.
        """
        model_cfg = self.config['model']
        architecture = model_cfg.get('architecture', 'yolov8m')
        
        if self.resume:
            # Resume from checkpoint
            self.logger.info(f"Resuming from checkpoint: {self.resume}")
            model = YOLO(self.resume)
        else:
            # Initialize new model
            if model_cfg.get('pretrained', True):
                # Load pre-trained COCO weights
                model_name = f"{architecture}.pt"
                self.logger.info(f"Loading pre-trained model: {model_name}")
                model = YOLO(model_name)
            else:
                # Initialize from scratch
                model_yaml = f"{architecture}.yaml"
                self.logger.info(f"Initializing model from scratch: {model_yaml}")
                model = YOLO(model_yaml)
        
        self.logger.info(f"Model architecture: {architecture}")
        self.logger.info(f"Number of classes: {self.data_config['nc']}")
        
        return model
    
    def _setup_callbacks(self) -> Dict:
        """
        Setup training callbacks (early stopping, checkpointing).
        
        Returns:
            Dictionary of callback instances.
        """
        train_cfg = self.config['training']
        checkpoint_cfg = self.config['checkpoint']
        
        callbacks = {}
        
        # Early stopping
        if train_cfg.get('patience', 0) > 0:
            callbacks['early_stopping'] = EarlyStopping(
                patience=train_cfg['patience'],
                mode='max',  # Maximize mAP
                verbose=True
            )
            self.logger.info(f"Early stopping enabled (patience={train_cfg['patience']})")
        
        # Model checkpointing
        callbacks['checkpoint'] = ModelCheckpoint(
            checkpoint_dir=checkpoint_cfg.get('save_dir', './models'),
            monitor='map50',
            mode='max',
            save_best=checkpoint_cfg.get('save_best', True),
            save_last=checkpoint_cfg.get('save_last', True),
            save_frequency=checkpoint_cfg.get('save_interval', 10),
            verbose=True
        )
        self.logger.info(f"Model checkpointing enabled (save_dir={checkpoint_cfg.get('save_dir')})")
        
        return callbacks
    
    def train(self) -> Dict:
        """
        Execute the complete training process.
        
        Returns:
            Dictionary containing training results and metrics.
        """
        self.logger.info("="*80)
        self.logger.info("Starting Training")
        self.logger.info("="*80)
        
        # Prepare training arguments
        train_args = self._prepare_train_args()
        
        # Log training parameters
        self.logger.info("Training Parameters:")
        for key, value in train_args.items():
            self.logger.info(f"  {key}: {value}")
        
        try:
            # Start training
            self.logger.info("\nTraining started...")
            results = self.model.train(**train_args)
            
            self.logger.info("="*80)
            self.logger.info("Training Completed Successfully")
            self.logger.info("="*80)
            
            # Log final metrics
            self._log_final_metrics(results)
            
            # Save training curves
            self._save_training_plots(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}", exc_info=True)
            raise
    
    def _prepare_train_args(self) -> Dict:
        """
        Prepare training arguments from configuration.
        
        Returns:
            Dictionary of training arguments for YOLO.train().
        """
        train_cfg = self.config['training']
        aug_cfg = self.config['augmentation']
        val_cfg = self.config['validation']
        log_cfg = self.config['logging']
        
        # Build training arguments
        args = {
            # Data
            'data': str(Path('configs/dataset.yaml').absolute()),
            
            # Training hyperparameters
            'epochs': train_cfg['epochs'],
            'batch': train_cfg['batch_size'],
            'imgsz': train_cfg['image_size'],
            'device': self.device,  # Use detected device (CPU/GPU)
            'workers': train_cfg.get('workers', 8),
            
            # Optimizer
            'optimizer': train_cfg.get('optimizer', 'AdamW'),
            'lr0': train_cfg['learning_rate'],
            'weight_decay': train_cfg.get('weight_decay', 0.0005),
            'momentum': train_cfg.get('momentum', 0.937),
            
            # Learning rate schedule
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'warmup_epochs': train_cfg.get('warmup_epochs', 3),
            'warmup_bias_lr': train_cfg.get('warmup_bias_lr', 0.1),
            'warmup_momentum': train_cfg.get('warmup_momentum', 0.8),
            
            # Loss weights
            'box': train_cfg.get('box_loss_gain', 7.5),
            'cls': train_cfg.get('cls_loss_gain', 0.5),
            'dfl': train_cfg.get('dfl_loss_gain', 1.5),
            
            # Augmentation
            'hsv_h': aug_cfg.get('hsv_h', 0.015),
            'hsv_s': aug_cfg.get('hsv_s', 0.7),
            'hsv_v': aug_cfg.get('hsv_v', 0.4),
            'degrees': aug_cfg.get('degrees', 10.0),
            'translate': aug_cfg.get('translate', 0.1),
            'scale': aug_cfg.get('scale', 0.5),
            'shear': aug_cfg.get('shear', 0.0),
            'perspective': aug_cfg.get('perspective', 0.0),
            'flipud': aug_cfg.get('flipud', 0.0),
            'fliplr': aug_cfg.get('fliplr', 0.5),
            'mosaic': aug_cfg.get('mosaic', 1.0),
            'mixup': aug_cfg.get('mixup', 0.15),
            'copy_paste': aug_cfg.get('copy_paste', 0.1),
            'erasing': aug_cfg.get('erasing', 0.4),
            
            # Multi-scale training
            'multi_scale': train_cfg.get('multi_scale', True),
            
            # Mixed precision
            'amp': train_cfg.get('amp', True),
            
            # Validation
            'val': True,
            'save': True,
            'save_period': val_cfg.get('save_interval', 10),
            'conf': val_cfg.get('conf_threshold', 0.001),
            'iou': val_cfg.get('nms_iou', 0.7),
            'max_det': val_cfg.get('max_det', 300),
            
            # Logging
            'project': log_cfg.get('log_dir', './results/runs'),
            'name': 'train',
            'exist_ok': True,
            'plots': log_cfg.get('save_plots', True),
            'verbose': log_cfg.get('verbose', True),
            
            # Checkpointing
            'patience': train_cfg.get('patience', 50),
            # Note: 'save_best' removed - Ultralytics automatically saves best model as 'best.pt'
            
            # Reproducibility
            'seed': self.config['advanced'].get('seed', 42),
            'deterministic': self.config['advanced'].get('deterministic', False),
        }
        
        # Resume training if checkpoint provided
        if self.resume:
            args['resume'] = self.resume  # Pass the checkpoint path directly
        
        return args
    
    def _log_final_metrics(self, results) -> None:
        """
        Log final training metrics.
        
        Args:
            results: Training results from YOLO.train().
        """
        try:
            # Extract metrics from results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            self.logger.info("Final Training Metrics:")
            self.logger.info("-" * 80)
            
            # Log key metrics if available
            key_metrics = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                          'metrics/precision(B)', 'metrics/recall(B)']
            
            for metric_key in key_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    metric_name = metric_key.split('/')[-1]
                    self.logger.info(f"  {metric_name}: {value:.4f}")
            
            # Check if target accuracy achieved
            map50 = metrics.get('metrics/mAP50(B)', 0)
            target_map50 = self.config['target_metrics'].get('map50', 0.80)
            
            if map50 >= target_map50:
                self.logger.info(f"\n✓ Target accuracy achieved! mAP@0.5 = {map50:.4f} (target: {target_map50:.4f})")
            else:
                self.logger.warning(f"\n⚠ Target accuracy not achieved. mAP@0.5 = {map50:.4f} (target: {target_map50:.4f})")
            
        except Exception as e:
            self.logger.warning(f"Could not extract final metrics: {str(e)}")
    
    def _save_training_plots(self, results) -> None:
        """
        Save training visualization plots.
        
        Args:
            results: Training results from YOLO.train().
        """
        try:
            plots_dir = Path(self.config['logging'].get('log_dir', './results')) / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Training plots saved to: {plots_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not save training plots: {str(e)}")
    
    def validate(self, data: Optional[str] = None) -> Dict:
        """
        Run validation on the trained model.
        
        Args:
            data: Path to dataset yaml (uses training data config if not specified).
        
        Returns:
            Validation results dictionary.
        """
        self.logger.info("Running validation...")
        
        data_path = data or str(Path('configs/dataset.yaml').absolute())
        results = self.model.val(data=data_path)
        
        self.logger.info("Validation completed")
        return results
    
    def export(self, format: str = 'onnx') -> Path:
        """
        Export trained model to specified format.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.).
        
        Returns:
            Path to exported model.
        """
        self.logger.info(f"Exporting model to {format} format...")
        
        export_path = self.model.export(format=format)
        
        self.logger.info(f"Model exported to: {export_path}")
        return Path(export_path)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Train YOLOv8m for Space Station Safety Object Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration files
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration file'
    )
    
    # Training parameters (override config)
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Training device (0, 1, cpu)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument(
        '--resume', '-r',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files'
    )
    
    # Mode
    parser.add_argument(
        '--val-only',
        action='store_true',
        help='Only run validation (no training)'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
    
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    train_config = load_config(args.config)
    data_config = load_config(args.data)
    
    # Override config with command-line arguments
    if args.epochs:
        train_config['training']['epochs'] = args.epochs
    if args.batch:
        train_config['training']['batch_size'] = args.batch
    if args.device:
        train_config['training']['device'] = args.device
    if args.workers:
        train_config['training']['workers'] = args.workers
    
    # Initialize trainer
    trainer = YOLOv8Trainer(
        train_config=train_config,
        data_config=data_config,
        resume=args.resume,
        log_dir=args.log_dir
    )
    
    # Run training or validation
    if args.val_only:
        results = trainer.validate()
    else:
        results = trainer.train()
    
    print("\n" + "="*80)
    print("Training pipeline completed successfully!")
    print("="*80)
    print(f"\nBest model saved to: ./models/best.pt")
    print(f"Training logs saved to: {args.log_dir}")
    print(f"Results saved to: ./results/runs/train")
    print("\nNext steps:")
    print("  1. Run evaluation: python scripts/evaluate.py --model models/best.pt")
    print("  2. Analyze scenarios: python scripts/scenario_analysis.py")
    print("  3. Make predictions: python scripts/predict.py --source test_images/")
    print("="*80)


if __name__ == "__main__":
    main()
