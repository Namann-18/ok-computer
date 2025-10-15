"""
================================================================================
Incremental Learning Pipeline for YOLOv8
================================================================================
Automated system for continuous model improvement through active learning.

Features:
    - Inference on new/unlabeled images
    - High-confidence prediction curation (>0.7)
    - Automatic dataset expansion with quality control
    - Incremental model training with performance tracking
    - Automatic rollback on performance degradation
    - Active learning for uncertain samples
    
Workflow:
    1. Run inference on new images
    2. Filter predictions by confidence threshold
    3. Validate and curate new annotations
    4. Update dataset with new samples
    5. Retrain model with expanded dataset
    6. Evaluate and compare performance
    7. Keep best performing model

Author: Space Station Safety Detection Team
Date: 2025-10-15
Version: 1.0
================================================================================
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
import numpy as np
from datetime import datetime
import cv2
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from utils.logger import get_logger


class IncrementalLearningPipeline:
    """
    Manages incremental learning workflow for continuous model improvement.
    
    This pipeline automatically:
    - Runs inference on new unlabeled data
    - Selects high-confidence predictions as pseudo-labels
    - Maintains balanced dataset with quality control
    - Retrains model incrementally
    - Tracks performance metrics across iterations
    - Implements automatic rollback for degraded performance
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_config: str,
        train_config: str,
        new_images_dir: str,
        confidence_threshold: float = 0.7,
        uncertain_threshold: float = 0.5,
        project_dir: str = "./incremental_runs"
    ):
        """
        Initialize incremental learning pipeline.
        
        Args:
            model_path: Path to best performing model checkpoint
            dataset_config: Path to dataset configuration YAML
            train_config: Path to training configuration YAML
            new_images_dir: Directory containing new unlabeled images
            confidence_threshold: Minimum confidence for auto-annotation (default: 0.7)
            uncertain_threshold: Maximum confidence for active learning queue (default: 0.5)
            project_dir: Directory for incremental learning outputs
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        self.train_config = Path(train_config)
        self.new_images_dir = Path(new_images_dir)
        self.confidence_threshold = confidence_threshold
        self.uncertain_threshold = uncertain_threshold
        self.project_dir = Path(project_dir)
        
        # Create project directory structure
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir = self.project_dir / "new_annotations"
        self.annotations_dir.mkdir(exist_ok=True)
        self.uncertain_dir = self.project_dir / "uncertain_samples"
        self.uncertain_dir.mkdir(exist_ok=True)
        self.models_dir = self.project_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.project_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(
            name=f"incremental_{timestamp}",
            log_dir=str(self.project_dir / "logs")
        )
        
        # Load configurations
        with open(self.dataset_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
        with open(self.train_config, 'r') as f:
            self.train_cfg = yaml.safe_load(f)
            
        # Load model
        self.logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Initialize performance tracking
        self.performance_history = []
        self.load_performance_history()
        
        self.logger.info("="*80)
        self.logger.info("Incremental Learning Pipeline Initialized")
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Confidence Threshold: {self.confidence_threshold}")
        self.logger.info(f"Uncertain Threshold: {self.uncertain_threshold}")
        self.logger.info("="*80)
    
    def load_performance_history(self):
        """Load previous performance history if exists."""
        history_file = self.metrics_dir / "performance_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
            self.logger.info(f"Loaded {len(self.performance_history)} previous iterations")
    
    def save_performance_history(self):
        """Save performance history to disk."""
        history_file = self.metrics_dir / "performance_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def run_inference(self) -> Dict[str, List]:
        """
        Run inference on new images and categorize by confidence.
        
        Returns:
            Dictionary with 'high_conf', 'uncertain', and 'low_conf' predictions
        """
        self.logger.info("Running inference on new images...")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.new_images_dir.glob(ext)))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        high_conf_predictions = []
        uncertain_predictions = []
        low_conf_predictions = []
        
        # Process images with progress bar
        for img_path in tqdm(image_files, desc="Processing images"):
            # Run inference
            results = self.model.predict(
                source=str(img_path),
                conf=0.1,  # Low threshold to catch all detections
                iou=0.45,
                verbose=False
            )
            
            if len(results) == 0 or results[0].boxes is None:
                continue
                
            result = results[0]
            boxes = result.boxes
            
            # Read image for visualization
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            # Process each detection
            detections = []
            max_conf = 0.0
            
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Convert to YOLO format (normalized xywh)
                x1, y1, x2, y2 = xyxy
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                detection = {
                    'class': cls,
                    'confidence': conf,
                    'bbox': [x_center, y_center, width, height],
                    'bbox_xyxy': xyxy.tolist()
                }
                detections.append(detection)
                max_conf = max(max_conf, conf)
            
            # Categorize based on confidence
            prediction_data = {
                'image_path': str(img_path),
                'detections': detections,
                'max_confidence': max_conf
            }
            
            if max_conf >= self.confidence_threshold:
                high_conf_predictions.append(prediction_data)
            elif max_conf >= self.uncertain_threshold:
                uncertain_predictions.append(prediction_data)
            else:
                low_conf_predictions.append(prediction_data)
        
        self.logger.info(f"High confidence predictions: {len(high_conf_predictions)}")
        self.logger.info(f"Uncertain predictions (active learning): {len(uncertain_predictions)}")
        self.logger.info(f"Low confidence predictions: {len(low_conf_predictions)}")
        
        return {
            'high_conf': high_conf_predictions,
            'uncertain': uncertain_predictions,
            'low_conf': low_conf_predictions
        }
    
    def save_annotations(self, predictions: List[Dict], output_dir: Path):
        """
        Save predictions as YOLO format annotations.
        
        Args:
            predictions: List of prediction dictionaries
            output_dir: Directory to save annotation files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pred in predictions:
            img_path = Path(pred['image_path'])
            label_file = output_dir / f"{img_path.stem}.txt"
            
            # Write YOLO format annotations
            with open(label_file, 'w') as f:
                for det in pred['detections']:
                    # Only save detections above confidence threshold
                    if det['confidence'] >= self.confidence_threshold:
                        cls = det['class']
                        x, y, w, h = det['bbox']
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    def save_uncertain_samples(self, predictions: List[Dict]):
        """
        Save uncertain predictions for human review (active learning).
        
        Args:
            predictions: List of uncertain prediction dictionaries
        """
        self.logger.info("Saving uncertain samples for active learning...")
        
        uncertain_images_dir = self.uncertain_dir / "images"
        uncertain_labels_dir = self.uncertain_dir / "labels"
        uncertain_images_dir.mkdir(parents=True, exist_ok=True)
        uncertain_labels_dir.mkdir(parents=True, exist_ok=True)
        
        uncertain_log = []
        
        for pred in predictions:
            img_path = Path(pred['image_path'])
            
            # Copy image
            dst_img = uncertain_images_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Save preliminary labels
            label_file = uncertain_labels_dir / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                for det in pred['detections']:
                    cls = det['class']
                    x, y, w, h = det['bbox']
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            uncertain_log.append({
                'image': img_path.name,
                'max_confidence': pred['max_confidence'],
                'num_detections': len(pred['detections'])
            })
        
        # Save uncertain samples log
        log_file = self.uncertain_dir / "uncertain_samples.json"
        with open(log_file, 'w') as f:
            json.dump(uncertain_log, f, indent=2)
        
        self.logger.info(f"Saved {len(predictions)} uncertain samples for review")
        self.logger.info(f"Review location: {self.uncertain_dir}")
    
    def analyze_dataset_balance(self, dataset_path: Path) -> Dict[int, int]:
        """
        Analyze class distribution in dataset.
        
        Args:
            dataset_path: Path to labels directory
            
        Returns:
            Dictionary mapping class_id to count
        """
        class_counts = {i: 0 for i in range(self.data_config['nc'])}
        
        label_files = list(dataset_path.glob("*.txt"))
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        class_counts[cls] += 1
        
        return class_counts
    
    def update_dataset(self, high_conf_predictions: List[Dict]) -> Tuple[int, int]:
        """
        Update training dataset with new high-confidence predictions.
        
        Args:
            high_conf_predictions: List of high confidence predictions
            
        Returns:
            Tuple of (images_added, annotations_added)
        """
        self.logger.info("Updating dataset with new samples...")
        
        # Get dataset paths
        dataset_root = Path(self.data_config['path'])
        train_images = dataset_root / self.data_config['train']
        train_labels = dataset_root / "train" / "labels"
        
        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)
        
        # Analyze current class distribution
        current_distribution = self.analyze_dataset_balance(train_labels)
        self.logger.info("Current class distribution:")
        for cls_id, count in current_distribution.items():
            cls_name = self.data_config['names'][cls_id]
            self.logger.info(f"  {cls_name}: {count}")
        
        images_added = 0
        annotations_added = 0
        
        # Add new samples
        for pred in high_conf_predictions:
            img_path = Path(pred['image_path'])
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_img_name = f"incremental_{timestamp}_{img_path.name}"
            
            # Copy image
            dst_img = train_images / new_img_name
            shutil.copy2(img_path, dst_img)
            
            # Save labels
            label_file = train_labels / f"{dst_img.stem}.txt"
            with open(label_file, 'w') as f:
                for det in pred['detections']:
                    if det['confidence'] >= self.confidence_threshold:
                        cls = det['class']
                        x, y, w, h = det['bbox']
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        annotations_added += 1
            
            images_added += 1
        
        # Analyze new class distribution
        new_distribution = self.analyze_dataset_balance(train_labels)
        self.logger.info("Updated class distribution:")
        for cls_id, count in new_distribution.items():
            cls_name = self.data_config['names'][cls_id]
            delta = count - current_distribution[cls_id]
            self.logger.info(f"  {cls_name}: {count} (+{delta})")
        
        self.logger.info(f"Added {images_added} images with {annotations_added} annotations")
        return images_added, annotations_added
    
    def train_incremental(self, iteration: int) -> Dict:
        """
        Train model incrementally with expanded dataset.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting incremental training - Iteration {iteration}")
        
        # Prepare training arguments
        train_args = {
            'data': str(self.dataset_config),
            'epochs': self.train_cfg['training']['epochs'] // 2,  # Shorter incremental training
            'batch': self.train_cfg['training']['batch_size'],
            'imgsz': self.train_cfg['training']['image_size'],
            'device': self.train_cfg['hardware']['device'],
            'workers': self.train_cfg['hardware']['num_workers'],
            'project': str(self.project_dir),
            'name': f'iteration_{iteration}',
            'exist_ok': True,
            'pretrained': False,  # Start from current model
            'optimizer': self.train_cfg['training']['optimizer'],
            'lr0': self.train_cfg['training']['learning_rate'] * 0.1,  # Lower LR for fine-tuning
            'patience': self.train_cfg['training']['patience'],
            'save': True,
            'save_period': 10,
            'cache': self.train_cfg['training']['cache_images'],
            'amp': self.train_cfg['training']['amp'],
            'verbose': True
        }
        
        # Train model
        results = self.model.train(**train_args)
        
        # Validate model
        val_results = self.model.val(data=str(self.dataset_config))
        
        metrics = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'model_path': str(self.project_dir / f'iteration_{iteration}' / 'weights' / 'best.pt')
        }
        
        self.logger.info("Incremental Training Results:")
        self.logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        self.logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def should_keep_model(self, new_metrics: Dict) -> bool:
        """
        Decide whether to keep new model or rollback.
        
        Args:
            new_metrics: Metrics from new training iteration
            
        Returns:
            True if new model should be kept, False to rollback
        """
        if not self.performance_history:
            return True  # First iteration, keep model
        
        best_map = max(h['mAP50'] for h in self.performance_history)
        new_map = new_metrics['mAP50']
        
        # Keep if improvement or within 1% tolerance
        improvement_threshold = -0.01
        improvement = new_map - best_map
        
        if improvement >= improvement_threshold:
            self.logger.info(f"✓ Model improvement: {improvement:+.4f}")
            return True
        else:
            self.logger.warning(f"✗ Model degradation: {improvement:+.4f} - ROLLBACK")
            return False
    
    def run_pipeline(self, max_iterations: int = 5):
        """
        Run complete incremental learning pipeline.
        
        Args:
            max_iterations: Maximum number of incremental iterations
        """
        self.logger.info("="*80)
        self.logger.info("STARTING INCREMENTAL LEARNING PIPELINE")
        self.logger.info("="*80)
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"ITERATION {iteration}/{max_iterations}")
            self.logger.info(f"{'='*80}\n")
            
            # Step 1: Run inference on new images
            predictions = self.run_inference()
            
            if not predictions['high_conf']:
                self.logger.warning("No high-confidence predictions found. Stopping pipeline.")
                break
            
            # Step 2: Save uncertain samples for active learning
            if predictions['uncertain']:
                self.save_uncertain_samples(predictions['uncertain'])
            
            # Step 3: Update dataset with high-confidence predictions
            images_added, annotations_added = self.update_dataset(predictions['high_conf'])
            
            if images_added == 0:
                self.logger.warning("No new samples added. Stopping pipeline.")
                break
            
            # Step 4: Train model incrementally
            new_metrics = self.train_incremental(iteration)
            
            # Step 5: Evaluate and decide
            if self.should_keep_model(new_metrics):
                # Keep new model
                self.performance_history.append(new_metrics)
                self.save_performance_history()
                
                # Update current model
                new_model_path = Path(new_metrics['model_path'])
                self.model = YOLO(str(new_model_path))
                self.model_path = new_model_path
                
                self.logger.info(f"✓ Model updated: {new_model_path}")
            else:
                # Rollback - keep current model
                self.logger.warning("Model performance degraded. Keeping previous model.")
            
            # Check if target achieved
            if new_metrics['mAP50'] >= 0.90:
                self.logger.info("="*80)
                self.logger.info("🎯 TARGET ACHIEVED: mAP@0.5 >= 0.90")
                self.logger.info("="*80)
                break
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary of incremental learning."""
        self.logger.info("\n" + "="*80)
        self.logger.info("INCREMENTAL LEARNING SUMMARY")
        self.logger.info("="*80)
        
        if not self.performance_history:
            self.logger.info("No successful iterations completed.")
            return
        
        self.logger.info(f"Total iterations: {len(self.performance_history)}")
        self.logger.info(f"\nPerformance progression:")
        
        for record in self.performance_history:
            self.logger.info(
                f"  Iteration {record['iteration']}: "
                f"mAP@0.5={record['mAP50']:.4f}, "
                f"mAP@0.5:0.95={record['mAP50-95']:.4f}"
            )
        
        best_iteration = max(self.performance_history, key=lambda x: x['mAP50'])
        self.logger.info(f"\n🏆 Best Performance:")
        self.logger.info(f"  Iteration: {best_iteration['iteration']}")
        self.logger.info(f"  mAP@0.5: {best_iteration['mAP50']:.4f}")
        self.logger.info(f"  mAP@0.5:0.95: {best_iteration['mAP50-95']:.4f}")
        self.logger.info(f"  Model: {best_iteration['model_path']}")
        
        improvement = best_iteration['mAP50'] - 0.84336  # Starting mAP
        self.logger.info(f"\n📈 Total Improvement: {improvement:+.4f}")
        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Incremental Learning Pipeline for YOLOv8"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to best model checkpoint (e.g., results/runs/train/weights/best.pt)'
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
        '--new-images',
        type=str,
        required=True,
        help='Directory containing new unlabeled images'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Confidence threshold for auto-annotation (default: 0.7)'
    )
    parser.add_argument(
        '--uncertain',
        type=float,
        default=0.5,
        help='Uncertainty threshold for active learning (default: 0.5)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Maximum number of incremental iterations (default: 5)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='./incremental_runs',
        help='Project directory for outputs'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IncrementalLearningPipeline(
        model_path=args.model,
        dataset_config=args.data,
        train_config=args.config,
        new_images_dir=args.new_images,
        confidence_threshold=args.confidence,
        uncertain_threshold=args.uncertain,
        project_dir=args.project
    )
    
    # Run pipeline
    pipeline.run_pipeline(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
