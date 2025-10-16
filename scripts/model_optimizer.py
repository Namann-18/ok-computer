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
    
Goal: Push mAP@50 from 85.8% to 90%+ through systematic optimization

Author: Space Station Safety Detection Team  
Date: 2025-10-16
Version: 2.0 - Enhanced for 90% mAP@50 target with auto-save to final/
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
        target_score: float = 0.90,
        baseline_score: float = 0.858,  # Current best mAP@50
        final_dir: str = './final'
    ):
        """
        Initialize optimizer.
        
        Args:
            model_path: Path to base model
            dataset_config: Path to dataset configuration
            train_config: Path to training configuration
            optimization_goal: Metric to optimize ('map50', 'map', 'precision', 'recall')
            target_score: Target metric score (default: 0.90 for 90% mAP@50)
            baseline_score: Current baseline score to beat (default: 0.858)
            final_dir: Directory to save models that beat baseline
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        self.train_config = Path(train_config)
        self.optimization_goal = optimization_goal
        self.target_score = target_score
        self.baseline_score = baseline_score
        self.final_dir = Path(final_dir)
        
        # Create final directory if it doesn't exist
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.logger.info("YOLOv8 Model Optimizer v2.0 - Enhanced Edition")
        self.logger.info(f"Base model: {self.model_path}")
        self.logger.info(f"Current baseline: {baseline_score:.4f} mAP@50")
        self.logger.info(f"Optimization goal: {optimization_goal}")
        self.logger.info(f"Target score: {target_score:.4f} (90% mAP@50)")
        self.logger.info(f"Improvement needed: {(target_score - baseline_score):.4f}")
        self.logger.info(f"Final models directory: {self.final_dir}")
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
                    'epochs': 500,  # Increased from 400 for better convergence
                    'patience': 100,  # More patience
                    'close_mosaic': 20,  # Disable mosaic in last N epochs
                }
            },
            'strategy_5_ultra_extended': {
                'name': 'Ultra Extended Training',
                'description': 'Maximum epochs for 90% target',
                'params': {
                    'epochs': 600,  # Maximum training
                    'patience': 120,
                    'close_mosaic': 30,
                    'lr0': 0.0008,  # Higher initial LR
                    'lrf': 0.005,   # Lower final LR multiplier
                    'warmup_epochs': 10,
                    'momentum': 0.96,
                    'weight_decay': 0.001,
                }
            },
            'strategy_6_advanced_aug': {
                'name': 'Advanced Augmentation + Extended',
                'description': 'Combine aggressive aug with extended training',
                'params': {
                    'epochs': 500,
                    'patience': 100,
                    'hsv_h': 0.025,
                    'hsv_s': 0.9,
                    'hsv_v': 0.6,
                    'degrees': 25.0,
                    'translate': 0.25,
                    'scale': 0.95,
                    'shear': 12.0,
                    'flipud': 0.25,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.4,
                    'copy_paste': 0.4,
                    'close_mosaic': 25,
                }
            },
            'strategy_7_loss_tuned': {
                'name': 'Fine-Tuned Loss Functions',
                'description': 'Optimized loss weights for 90% target',
                'params': {
                    'epochs': 500,
                    'patience': 100,
                    'box': 8.0,  # Higher box loss
                    'cls': 1.2,  # Higher class loss
                    'dfl': 2.0,  # Higher DFL loss
                    'lr0': 0.0007,
                    'lrf': 0.008,
                }
            },
            'strategy_8_mega_batch': {
                'name': 'Large Batch Extended Training',
                'description': 'Larger batch with extended epochs',
                'params': {
                    'epochs': 550,
                    'batch': 32,  # Larger batch if GPU allows
                    'patience': 110,
                    'lr0': 0.001,  # Higher LR for larger batch
                    'close_mosaic': 30,
                }
            },
            'strategy_fast_boost': {
                'name': 'Fast Aggressive Boost (2hr)',
                'description': 'Maximum impact in 60 epochs - aggressive everything',
                'params': {
                    'epochs': 60,
                    'patience': 15,
                    'lr0': 0.0015,  # Higher LR for faster convergence
                    'lrf': 0.01,
                    'warmup_epochs': 3,
                    'momentum': 0.97,
                    'weight_decay': 0.0012,
                    'box': 9.0,  # Very high box loss
                    'cls': 1.5,  # Very high cls loss
                    'dfl': 2.5,  # Very high DFL
                    'hsv_h': 0.03,
                    'hsv_s': 1.0,
                    'hsv_v': 0.7,
                    'degrees': 30.0,
                    'translate': 0.3,
                    'scale': 1.0,
                    'mixup': 0.5,
                    'copy_paste': 0.5,
                    'close_mosaic': 5,
                }
            },
            'strategy_quick_tune': {
                'name': 'Quick Fine-Tune (2hr)',
                'description': 'Fast fine-tuning with optimized losses',
                'params': {
                    'epochs': 50,
                    'patience': 12,
                    'lr0': 0.001,
                    'lrf': 0.005,
                    'warmup_epochs': 2,
                    'box': 10.0,  # Maximum box loss for precision
                    'cls': 2.0,   # Maximum class loss
                    'dfl': 3.0,   # Maximum DFL
                    'momentum': 0.98,
                    'weight_decay': 0.0015,
                }
            },
            'strategy_rapid_convergence': {
                'name': 'Rapid Convergence (2hr)',
                'description': 'Fast convergence with high learning rate',
                'params': {
                    'epochs': 45,
                    'patience': 10,
                    'lr0': 0.002,  # Very high initial LR
                    'lrf': 0.001,
                    'warmup_epochs': 2,
                    'momentum': 0.99,
                    'weight_decay': 0.002,
                    'box': 8.5,
                    'cls': 1.8,
                    'dfl': 2.2,
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
        
        # Check if this beats the baseline
        if metrics['mAP50'] > self.baseline_score:
            improvement = metrics['mAP50'] - self.baseline_score
            self.logger.info(f"\n🎉 NEW RECORD! Improvement: +{improvement:.4f}")
            self.logger.info(f"  Previous best: {self.baseline_score:.4f}")
            self.logger.info(f"  New best: {metrics['mAP50']:.4f}")
            
            # Copy best model to final directory
            import shutil
            source_model = Path(metrics['model_path'])
            if source_model.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                final_model_name = f"best_mAP{metrics['mAP50']:.4f}_{strategy_name}_{timestamp}.pt"
                final_model_path = self.final_dir / final_model_name
                
                shutil.copy2(source_model, final_model_path)
                self.logger.info(f"\n✅ Model saved to: {final_model_path}")
                
                # Also save as 'best.pt' (overwrite previous best)
                best_model_path = self.final_dir / 'best.pt'
                shutil.copy2(source_model, best_model_path)
                self.logger.info(f"✅ Updated: {best_model_path}")
                
                # Save metrics
                metrics_file = self.final_dir / f"metrics_{timestamp}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                self.logger.info(f"✅ Metrics saved: {metrics_file}")
                
                # Update baseline for next iteration
                self.baseline_score = metrics['mAP50']
        else:
            self.logger.info(f"\n📊 No improvement over baseline ({self.baseline_score:.4f})")
        
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
            
            improvement = best_result['mAP50'] - 0.858  # Current baseline
            self.logger.info(f"\n📈 Total Improvement: {improvement:+.4f}")
            
            if best_result['mAP50'] >= self.target_score:
                self.logger.info(f"\n🎯 TARGET ACHIEVED! {best_result['mAP50']:.4f} >= {self.target_score:.4f}")
            else:
                remaining = self.target_score - best_result['mAP50']
                self.logger.info(f"\n📊 Progress: {best_result['mAP50']:.4f}/{self.target_score:.4f}")
                self.logger.info(f"   Still needed: {remaining:.4f} to reach 90%")
            
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
        help='Target mAP@50 score (default: 0.90 for 90%%)'
    )
    parser.add_argument(
        '--baseline',
        type=float,
        default=0.858,
        help='Current baseline mAP@50 to beat (default: 0.858)'
    )
    parser.add_argument(
        '--final-dir',
        type=str,
        default='./final',
        help='Directory to save improved models (default: ./final)'
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
        target_score=args.target,
        baseline_score=args.baseline,
        final_dir=args.final_dir
    )
    
    # Run optimization
    optimizer.optimize(strategies_to_try=args.strategies)


if __name__ == "__main__":
    main()
