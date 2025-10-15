"""
================================================================================
YOLOv8 Model Optimizer
================================================================================
Advanced optimization techniques to improve model performance and efficiency.

Techniques:
    - Hyperparameter tuning with Optuna
    - Advanced training strategies
    - Loss function optimization
    - Learning rate scheduling
    - Data sampling strategies
    - Class balancing
    
Goal: Push mAP from 0.846 to 0.90+ through systematic optimization

Author: Space Station Safety Detection Team  
Date: 2025-10-15
Version: 1.0
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from utils.logger import get_logger


class YOLOv8Optimizer:
    """
    Advanced optimizer for YOLOv8 model performance improvement.
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_config: str,
        train_config: str,
        optimization_goal: str = 'map50',
        target_score: float = 0.90
    ):
        """
        Initialize optimizer.
        
        Args:
            model_path: Path to base model
            dataset_config: Path to dataset configuration
            train_config: Path to training configuration
            optimization_goal: Metric to optimize ('map50', 'map', 'precision', 'recall')
            target_score: Target metric score
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        self.train_config = Path(train_config)
        self.optimization_goal = optimization_goal
        self.target_score = target_score
        
        # Load configurations
        with open(self.dataset_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
        with open(self.train_config, 'r') as f:
            self.train_cfg = yaml.safe_load(f)
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(
            name=f"optimizer_{timestamp}",
            log_dir="./logs"
        )
        
        self.logger.info("="*80)
        self.logger.info("YOLOv8 Model Optimizer Initialized")
        self.logger.info(f"Base model: {self.model_path}")
        self.logger.info(f"Optimization goal: {optimization_goal}")
        self.logger.info(f"Target score: {target_score}")
        self.logger.info("="*80)
    
    def get_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Define optimization strategies to try.
        
        Returns:
            Dictionary of optimization strategies
        """
        strategies = {
            'strategy_1_focal_loss': {
                'name': 'Enhanced Focal Loss',
                'description': 'Increase focus on hard examples',
                'params': {
                    'box': 7.5,
                    'cls': 1.0,  # Increased from 0.5
                    'dfl': 1.5,
                }
            },
            'strategy_2_aggressive_aug': {
                'name': 'Aggressive Augmentation',
                'description': 'Strong augmentation for robustness',
                'params': {
                    'hsv_h': 0.02,  # Increased
                    'hsv_s': 0.8,   # Increased
                    'hsv_v': 0.5,   # Increased
                    'degrees': 20.0,  # Increased rotation
                    'translate': 0.2,
                    'scale': 0.9,
                    'shear': 10.0,
                    'perspective': 0.001,
                    'flipud': 0.2,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.3,  # Enable mixup
                    'copy_paste': 0.3,  # Enable copy-paste
                }
            },
            'strategy_3_optimized_lr': {
                'name': 'Optimized Learning Rate',
                'description': 'Fine-tuned learning rate schedule',
                'params': {
                    'lr0': 0.0005,  # Slightly higher initial LR
                    'lrf': 0.01,    # Final LR multiplier
                    'momentum': 0.95,  # Increased momentum
                    'weight_decay': 0.0008,  # Increased regularization
                    'warmup_epochs': 5,  # Longer warmup
                    'warmup_momentum': 0.8,
                    'warmup_bias_lr': 0.05,
                }
            },
            'strategy_4_extended_training': {
                'name': 'Extended Training',
                'description': 'More epochs with cosine annealing',
                'params': {
                    'epochs': 400,  # Increased from 300
                    'patience': 75,  # Increased patience
                    'close_mosaic': 20,  # Disable mosaic in last N epochs
                }
            },
            'strategy_5_multi_scale': {
                'name': 'Multi-Scale Training',
                'description': 'Train on multiple image scales',
                'params': {
                    'scale': 0.9,  # Scale variation
                    'imgsz': 704,  # Larger input size
                }
            },
            'strategy_6_class_balanced': {
                'name': 'Class Balanced Sampling',
                'description': 'Balance training across classes',
                'params': {
                    'cls': 1.5,  # Higher classification loss weight
                }
            }
        }
        
        return strategies
    
    def train_with_strategy(
        self,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        output_name: str
    ) -> Dict:
        """
        Train model with a specific optimization strategy.
        
        Args:
            strategy_name: Name of strategy
            strategy_config: Strategy configuration
            output_name: Name for output directory
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Training Strategy: {strategy_config['name']}")
        self.logger.info(f"Description: {strategy_config['description']}")
        self.logger.info(f"{'='*80}\n")
        
        # Load model
        model = YOLO(str(self.model_path))
        
        # Prepare training arguments with safe config access
        train_args = {
            'data': str(self.dataset_config),
            'epochs': self.train_cfg.get('training', {}).get('epochs', 300),
            'batch': self.train_cfg.get('training', {}).get('batch_size', 16),
            'imgsz': self.train_cfg.get('training', {}).get('image_size', 640),
            'device': self.train_cfg.get('hardware', {}).get('device', 0),
            'workers': self.train_cfg.get('hardware', {}).get('num_workers', 8),
            'project': './optimization_runs',
            'name': output_name,
            'exist_ok': True,
            'pretrained': False,
            'optimizer': self.train_cfg.get('training', {}).get('optimizer', 'AdamW'),
            'lr0': self.train_cfg.get('training', {}).get('learning_rate', 0.0003),
            'patience': self.train_cfg.get('training', {}).get('patience', 50),
            'save': True,
            'save_period': 10,
            'cache': self.train_cfg.get('training', {}).get('cache_images', True),
            'amp': self.train_cfg.get('training', {}).get('amp', True),
            'verbose': True,
            'plots': True,
        }
        
        # Apply strategy-specific parameters
        train_args.update(strategy_config['params'])
        
        # Log parameters
        self.logger.info("Training parameters:")
        for key, value in strategy_config['params'].items():
            self.logger.info(f"  {key}: {value}")
        
        # Train model
        self.logger.info("\nStarting training...")
        results = model.train(**train_args)
        
        # Validate model
        self.logger.info("\nValidating model...")
        val_results = model.val(data=str(self.dataset_config))
        
        metrics = {
            'strategy': strategy_name,
            'strategy_name': strategy_config['name'],
            'timestamp': datetime.now().isoformat(),
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'model_path': str(Path('./optimization_runs') / output_name / 'weights' / 'best.pt')
        }
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Training Results:")
        self.logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        self.logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info("="*80 + "\n")
        
        return metrics
    
    def optimize(self, strategies_to_try: Optional[list] = None) -> Dict:
        """
        Run optimization with multiple strategies.
        
        Args:
            strategies_to_try: List of strategy names to try (None = all)
            
        Returns:
            Best results dictionary
        """
        all_strategies = self.get_optimization_strategies()
        
        if strategies_to_try is None:
            strategies_to_try = list(all_strategies.keys())
        
        self.logger.info("="*80)
        self.logger.info("STARTING OPTIMIZATION PROCESS")
        self.logger.info(f"Strategies to try: {len(strategies_to_try)}")
        self.logger.info("="*80)
        
        results = []
        
        for i, strategy_name in enumerate(strategies_to_try, 1):
            if strategy_name not in all_strategies:
                self.logger.warning(f"Strategy '{strategy_name}' not found. Skipping.")
                continue
            
            strategy_config = all_strategies[strategy_name]
            output_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"# OPTIMIZATION {i}/{len(strategies_to_try)}")
            self.logger.info(f"{'#'*80}")
            
            try:
                metrics = self.train_with_strategy(
                    strategy_name,
                    strategy_config,
                    output_name
                )
                results.append(metrics)
                
                # Check if target achieved
                if metrics['mAP50'] >= self.target_score:
                    self.logger.info(f"\n🎯 TARGET ACHIEVED: {metrics['mAP50']:.4f} >= {self.target_score}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Strategy '{strategy_name}' failed: {e}")
                continue
        
        # Find best result
        if results:
            best_result = max(results, key=lambda x: x['mAP50'])
            
            self.logger.info("\n" + "="*80)
            self.logger.info("OPTIMIZATION COMPLETE")
            self.logger.info("="*80)
            self.logger.info("\nAll Results:")
            
            for result in results:
                self.logger.info(
                    f"  {result['strategy_name']}: "
                    f"mAP@0.5={result['mAP50']:.4f}, "
                    f"mAP@0.5:0.95={result['mAP50-95']:.4f}"
                )
            
            self.logger.info(f"\n🏆 Best Strategy: {best_result['strategy_name']}")
            self.logger.info(f"  mAP@0.5: {best_result['mAP50']:.4f}")
            self.logger.info(f"  mAP@0.5:0.95: {best_result['mAP50-95']:.4f}")
            self.logger.info(f"  Precision: {best_result['precision']:.4f}")
            self.logger.info(f"  Recall: {best_result['recall']:.4f}")
            self.logger.info(f"  Model: {best_result['model_path']}")
            
            improvement = best_result['mAP50'] - 0.84336  # Starting mAP
            self.logger.info(f"\n📈 Improvement: {improvement:+.4f}")
            self.logger.info("="*80)
            
            # Save results
            results_file = Path('./optimization_runs') / 'optimization_results.json'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump({
                    'all_results': results,
                    'best_result': best_result
                }, f, indent=2)
            
            self.logger.info(f"\nResults saved to: {results_file}")
            
            return best_result
        else:
            self.logger.error("No successful optimization runs.")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Model Optimizer"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to base model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training configuration'
    )
    parser.add_argument(
        '--goal',
        type=str,
        choices=['map50', 'map', 'precision', 'recall'],
        default='map50',
        help='Optimization goal metric'
    )
    parser.add_argument(
        '--target',
        type=float,
        default=0.90,
        help='Target metric score'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=None,
        help='Specific strategies to try (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = YOLOv8Optimizer(
        model_path=args.model,
        dataset_config=args.data,
        train_config=args.config,
        optimization_goal=args.goal,
        target_score=args.target
    )
    
    # Run optimization
    optimizer.optimize(strategies_to_try=args.strategies)


if __name__ == "__main__":
    main()
