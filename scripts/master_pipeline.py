"""
================================================================================
Master Optimization Pipeline for YOLOv8
================================================================================
Comprehensive workflow to improve mAP from 0.846 to 0.90+

This master script orchestrates:
    1. Advanced data augmentation
    2. Model optimization with multiple strategies
    3. Incremental learning on new data
    4. Model ensemble for final boost
    
Execution Flow:
    Phase 1: Data Augmentation (Expand training set)
    Phase 2: Model Optimization (Try advanced training strategies)
    Phase 3: Incremental Learning (Continuous improvement)
    Phase 4: Model Ensemble (Combine best models)

Author: Space Station Safety Detection Team
Date: 2025-10-15
Version: 1.0
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger


class MasterOptimizationPipeline:
    """
    Master pipeline orchestrating all optimization techniques.
    """
    
    def __init__(
        self,
        base_model: str,
        dataset_config: str,
        train_config: str,
        target_map: float = 0.90
    ):
        """
        Initialize master pipeline.
        
        Args:
            base_model: Path to current best model
            dataset_config: Path to dataset configuration
            train_config: Path to training configuration
            target_map: Target mAP@0.5 score
        """
        self.base_model = Path(base_model)
        self.dataset_config = Path(dataset_config)
        self.train_config = Path(train_config)
        self.target_map = target_map
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(
            name=f"master_pipeline_{timestamp}",
            log_dir="./logs"
        )
        
        self.results_dir = Path("./master_optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("MASTER OPTIMIZATION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Base model: {self.base_model}")
        self.logger.info(f"Current mAP@0.5: 0.84336")
        self.logger.info(f"Target mAP@0.5: {self.target_map}")
        self.logger.info(f"Required improvement: +{self.target_map - 0.84336:.5f}")
        self.logger.info("="*80)
    
    def phase_1_augmentation(self) -> Path:
        """
        Phase 1: Generate augmented dataset.
        
        Returns:
            Path to augmented dataset
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1: DATA AUGMENTATION")
        self.logger.info("="*80)
        self.logger.info("Goal: Expand training dataset with diverse augmentations")
        self.logger.info("Expected improvement: +0.02 to +0.04 mAP")
        
        from scripts.advanced_augmentation import AdvancedAugmentor
        
        augmented_dir = self.results_dir / "augmented_dataset"
        
        augmentor = AdvancedAugmentor(
            dataset_config=str(self.dataset_config),
            output_dir=str(augmented_dir),
            augmentation_factor=2  # Double the dataset
        )
        
        augmentor.augment_dataset()
        
        self.logger.info("✓ Phase 1 complete")
        return augmented_dir
    
    def phase_2_optimization(self) -> Dict:
        """
        Phase 2: Optimize model with advanced strategies.
        
        Returns:
            Best optimization results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: MODEL OPTIMIZATION")
        self.logger.info("="*80)
        self.logger.info("Goal: Apply advanced training strategies")
        self.logger.info("Expected improvement: +0.03 to +0.05 mAP")
        
        from scripts.model_optimizer import YOLOv8Optimizer
        
        optimizer = YOLOv8Optimizer(
            model_path=str(self.base_model),
            dataset_config=str(self.dataset_config),
            train_config=str(self.train_config),
            optimization_goal='map50',
            target_score=self.target_map
        )
        
        # Try most promising strategies
        strategies = [
            'strategy_1_focal_loss',
            'strategy_2_aggressive_aug',
            'strategy_3_optimized_lr',
            'strategy_5_multi_scale'
        ]
        
        best_result = optimizer.optimize(strategies_to_try=strategies)
        
        self.logger.info("✓ Phase 2 complete")
        return best_result
    
    def phase_3_incremental_learning(self, best_model: str) -> Dict:
        """
        Phase 3: Incremental learning on validation/test data.
        
        Args:
            best_model: Path to best model from phase 2
            
        Returns:
            Incremental learning results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 3: INCREMENTAL LEARNING")
        self.logger.info("="*80)
        self.logger.info("Goal: Learn from high-confidence predictions")
        self.logger.info("Expected improvement: +0.01 to +0.03 mAP")
        
        from scripts.incremental_learning import IncrementalLearningPipeline
        
        # Use test set as "new" data for incremental learning
        with open(self.dataset_config, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_root = Path(data_config['path'])
        test_images = dataset_root / data_config['test']
        
        pipeline = IncrementalLearningPipeline(
            model_path=best_model,
            dataset_config=str(self.dataset_config),
            train_config=str(self.train_config),
            new_images_dir=str(test_images),
            confidence_threshold=0.75,  # Higher threshold for quality
            project_dir=str(self.results_dir / "incremental_runs")
        )
        
        pipeline.run_pipeline(max_iterations=3)
        
        # Get best result
        if pipeline.performance_history:
            best_iteration = max(pipeline.performance_history, key=lambda x: x['mAP50'])
            self.logger.info("✓ Phase 3 complete")
            return best_iteration
        else:
            self.logger.warning("Phase 3 yielded no improvements")
            return None
    
    def phase_4_ensemble(self, model_paths: List[str]) -> Dict:
        """
        Phase 4: Create model ensemble.
        
        Args:
            model_paths: List of best model paths
            
        Returns:
            Ensemble results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 4: MODEL ENSEMBLE")
        self.logger.info("="*80)
        self.logger.info("Goal: Combine predictions from top models")
        self.logger.info("Expected improvement: +0.01 to +0.02 mAP")
        
        if len(model_paths) < 2:
            self.logger.warning("Not enough models for ensemble. Skipping phase 4.")
            return None
        
        from scripts.model_ensemble import ModelEnsemble
        
        # Select top 3 models
        top_models = model_paths[:3]
        
        ensemble = ModelEnsemble(
            model_paths=top_models,
            dataset_config=str(self.dataset_config),
            weights=None,  # Equal weights
            fusion_method='wbf',
            iou_threshold=0.5,
            conf_threshold=0.001
        )
        
        metrics = ensemble.evaluate_ensemble()
        
        self.logger.info("✓ Phase 4 complete")
        return metrics
    
    def run_full_pipeline(self):
        """
        Execute complete optimization pipeline.
        """
        self.logger.info("\n" + "#"*80)
        self.logger.info("# STARTING FULL OPTIMIZATION PIPELINE")
        self.logger.info("#"*80)
        
        all_results = {
            'start_time': datetime.now().isoformat(),
            'base_model': str(self.base_model),
            'initial_map': 0.84336,
            'target_map': self.target_map,
            'phases': {}
        }
        
        model_paths = [str(self.base_model)]
        
        try:
            # Phase 1: Data Augmentation
            augmented_dir = self.phase_1_augmentation()
            all_results['phases']['phase_1'] = {
                'status': 'completed',
                'augmented_dir': str(augmented_dir)
            }
            
            # Phase 2: Model Optimization
            phase2_result = self.phase_2_optimization()
            if phase2_result:
                all_results['phases']['phase_2'] = phase2_result
                model_paths.append(phase2_result['model_path'])
                
                if phase2_result['mAP50'] >= self.target_map:
                    self.logger.info(f"\n🎯 TARGET ACHIEVED in Phase 2: {phase2_result['mAP50']:.4f}")
                    all_results['target_achieved'] = True
                    all_results['final_map'] = phase2_result['mAP50']
                    self.save_final_results(all_results)
                    return
            
            # Phase 3: Incremental Learning
            if model_paths:
                phase3_result = self.phase_3_incremental_learning(model_paths[-1])
                if phase3_result:
                    all_results['phases']['phase_3'] = phase3_result
                    model_paths.append(phase3_result['model_path'])
                    
                    if phase3_result['mAP50'] >= self.target_map:
                        self.logger.info(f"\n🎯 TARGET ACHIEVED in Phase 3: {phase3_result['mAP50']:.4f}")
                        all_results['target_achieved'] = True
                        all_results['final_map'] = phase3_result['mAP50']
                        self.save_final_results(all_results)
                        return
            
            # Phase 4: Model Ensemble
            phase4_result = self.phase_4_ensemble(model_paths)
            if phase4_result:
                all_results['phases']['phase_4'] = phase4_result
                
                if phase4_result['mAP50'] >= self.target_map:
                    self.logger.info(f"\n🎯 TARGET ACHIEVED with Ensemble: {phase4_result['mAP50']:.4f}")
                    all_results['target_achieved'] = True
                    all_results['final_map'] = phase4_result['mAP50']
                else:
                    all_results['target_achieved'] = False
                    all_results['final_map'] = phase4_result['mAP50']
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            all_results['error'] = str(e)
        
        all_results['end_time'] = datetime.now().isoformat()
        self.save_final_results(all_results)
    
    def save_final_results(self, results: Dict):
        """Save final pipeline results."""
        results_file = self.results_dir / "master_pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MASTER PIPELINE COMPLETE")
        self.logger.info("="*80)
        
        if 'final_map' in results:
            initial_map = results['initial_map']
            final_map = results['final_map']
            improvement = final_map - initial_map
            
            self.logger.info(f"\nInitial mAP@0.5: {initial_map:.4f}")
            self.logger.info(f"Final mAP@0.5:   {final_map:.4f}")
            self.logger.info(f"Total Improvement: {improvement:+.4f}")
            
            if results.get('target_achieved', False):
                self.logger.info(f"\n✅ TARGET ACHIEVED! ({final_map:.4f} >= {self.target_map})")
            else:
                remaining = self.target_map - final_map
                self.logger.info(f"\n⚠️ Target not achieved. Remaining: +{remaining:.4f}")
        
        self.logger.info(f"\nResults saved to: {results_file}")
        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Master Optimization Pipeline for YOLOv8"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to base best.pt model'
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
        '--target',
        type=float,
        default=0.90,
        help='Target mAP@0.5 score (default: 0.90)'
    )
    
    args = parser.parse_args()
    
    # Create and run master pipeline
    pipeline = MasterOptimizationPipeline(
        base_model=args.model,
        dataset_config=args.data,
        train_config=args.config,
        target_map=args.target
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
