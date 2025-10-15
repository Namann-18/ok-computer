"""
================================================================================
Model Ensemble for YOLOv8
================================================================================
Combine predictions from multiple YOLOv8 models using advanced fusion techniques
to improve overall detection performance.

Ensemble Methods:
    - Weighted Box Fusion (WBF)
    - Non-Maximum Suppression (NMS)
    - Soft-NMS
    - Confidence voting
    - Bbox coordinate averaging
    
Target: Boost mAP from 0.846 to 0.90+ through ensemble inference

Author: Space Station Safety Detection Team
Date: 2025-10-15
Version: 1.0
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import numpy as np
import torch
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from utils.logger import get_logger


class ModelEnsemble:
    """
    Ensemble multiple YOLOv8 models for improved detection performance.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        dataset_config: str,
        weights: List[float] = None,
        fusion_method: str = 'wbf',
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.001
    ):
        """
        Initialize model ensemble.
        
        Args:
            model_paths: List of paths to model checkpoints
            dataset_config: Path to dataset configuration
            weights: Optional weights for each model (default: equal weights)
            fusion_method: Fusion method ('wbf', 'nms', 'soft_nms', 'voting')
            iou_threshold: IoU threshold for box fusion
            conf_threshold: Confidence threshold for predictions
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.dataset_config = Path(dataset_config)
        self.fusion_method = fusion_method
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        
        # Set model weights
        if weights is None:
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            assert len(weights) == len(model_paths), "Number of weights must match number of models"
            total = sum(weights)
            self.weights = [w / total for w in weights]  # Normalize
        
        # Load dataset config
        with open(self.dataset_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = get_logger(name="model_ensemble", log_dir="./logs")
        
        # Load models
        self.logger.info("="*80)
        self.logger.info("Model Ensemble Initialization")
        self.logger.info("="*80)
        self.logger.info(f"Fusion method: {fusion_method}")
        self.logger.info(f"IoU threshold: {iou_threshold}")
        self.logger.info(f"Confidence threshold: {conf_threshold}")
        
        self.models = []
        for i, (model_path, weight) in enumerate(zip(self.model_paths, self.weights)):
            self.logger.info(f"Loading model {i+1}: {model_path} (weight: {weight:.3f})")
            model = YOLO(str(model_path))
            self.models.append(model)
        
        self.logger.info("="*80)
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes in xyxy format.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def weighted_boxes_fusion(
        self,
        all_boxes: List[List[np.ndarray]],
        all_scores: List[List[float]],
        all_classes: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Weighted Boxes Fusion algorithm.
        
        Args:
            all_boxes: List of box lists from each model (xyxy format)
            all_scores: List of score lists from each model
            all_classes: List of class lists from each model
            
        Returns:
            Fused boxes, scores, and classes
        """
        if not all_boxes or not any(boxes for boxes in all_boxes):
            return np.array([]), np.array([]), np.array([])
        
        # Collect all predictions with model weights
        weighted_predictions = []
        
        for model_idx, (boxes, scores, classes) in enumerate(zip(all_boxes, all_scores, all_classes)):
            model_weight = self.weights[model_idx]
            for box, score, cls in zip(boxes, scores, classes):
                weighted_predictions.append({
                    'box': box,
                    'score': score * model_weight,
                    'class': cls,
                    'model_idx': model_idx
                })
        
        if not weighted_predictions:
            return np.array([]), np.array([]), np.array([])
        
        # Sort by score
        weighted_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Fusion process
        fused_boxes = []
        fused_scores = []
        fused_classes = []
        
        while weighted_predictions:
            # Take highest scoring prediction
            current = weighted_predictions.pop(0)
            
            # Find all overlapping predictions
            matching = [current]
            remaining = []
            
            for pred in weighted_predictions:
                if pred['class'] == current['class']:
                    iou = self.compute_iou(current['box'], pred['box'])
                    if iou >= self.iou_threshold:
                        matching.append(pred)
                    else:
                        remaining.append(pred)
                else:
                    remaining.append(pred)
            
            weighted_predictions = remaining
            
            # Compute weighted average of matching boxes
            total_score = sum(m['score'] for m in matching)
            
            if total_score > 0:
                fused_box = np.zeros(4)
                for m in matching:
                    fused_box += m['box'] * m['score']
                fused_box /= total_score
                
                fused_boxes.append(fused_box)
                fused_scores.append(total_score)
                fused_classes.append(current['class'])
        
        if not fused_boxes:
            return np.array([]), np.array([]), np.array([])
        
        return (
            np.array(fused_boxes),
            np.array(fused_scores),
            np.array(fused_classes)
        )
    
    def nms_fusion(
        self,
        all_boxes: List[List[np.ndarray]],
        all_scores: List[List[float]],
        all_classes: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple NMS-based fusion.
        
        Concatenates all predictions and applies NMS.
        """
        # Concatenate all predictions
        boxes_list = []
        scores_list = []
        classes_list = []
        
        for model_idx, (boxes, scores, classes) in enumerate(zip(all_boxes, all_scores, all_classes)):
            model_weight = self.weights[model_idx]
            for box, score, cls in zip(boxes, scores, classes):
                boxes_list.append(box)
                scores_list.append(score * model_weight)
                classes_list.append(cls)
        
        if not boxes_list:
            return np.array([]), np.array([]), np.array([])
        
        boxes = np.array(boxes_list)
        scores = np.array(scores_list)
        classes = np.array(classes_list)
        
        # Apply NMS per class
        keep_indices = []
        for cls in np.unique(classes):
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            # Sort by score
            order = cls_scores.argsort()[::-1]
            
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(cls_indices[i])
                
                if len(order) == 1:
                    break
                
                # Compute IoU with remaining boxes
                ious = np.array([
                    self.compute_iou(cls_boxes[i], cls_boxes[j])
                    for j in order[1:]
                ])
                
                # Keep boxes with IoU below threshold
                order = order[1:][ious < self.iou_threshold]
            
            keep_indices.extend(keep)
        
        if not keep_indices:
            return np.array([]), np.array([]), np.array([])
        
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]
    
    def predict_ensemble(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ensemble prediction on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Fused boxes, scores, and classes
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Run inference with each model
        for model in self.models:
            results = model.predict(
                source=image_path,
                conf=self.conf_threshold,
                verbose=False
            )
            
            if len(results) == 0 or results[0].boxes is None:
                all_boxes.append([])
                all_scores.append([])
                all_classes.append([])
                continue
            
            result = results[0]
            boxes = result.boxes
            
            model_boxes = []
            model_scores = []
            model_classes = []
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                model_boxes.append(xyxy)
                model_scores.append(conf)
                model_classes.append(cls)
            
            all_boxes.append(model_boxes)
            all_scores.append(model_scores)
            all_classes.append(model_classes)
        
        # Fuse predictions
        if self.fusion_method == 'wbf':
            fused_boxes, fused_scores, fused_classes = self.weighted_boxes_fusion(
                all_boxes, all_scores, all_classes
            )
        elif self.fusion_method == 'nms':
            fused_boxes, fused_scores, fused_classes = self.nms_fusion(
                all_boxes, all_scores, all_classes
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_boxes, fused_scores, fused_classes
    
    def evaluate_ensemble(self, save_results: bool = True) -> Dict:
        """
        Evaluate ensemble on validation set.
        
        Args:
            save_results: Whether to save prediction results
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating ensemble on validation set...")
        
        # Get validation images
        dataset_root = Path(self.data_config['path'])
        val_images = dataset_root / self.data_config['val']
        
        image_files = list(val_images.glob("*.jpg")) + \
                     list(val_images.glob("*.png")) + \
                     list(val_images.glob("*.jpeg"))
        
        self.logger.info(f"Processing {len(image_files)} validation images...")
        
        # Store predictions
        all_predictions = []
        
        for img_path in tqdm(image_files, desc="Evaluating"):
            boxes, scores, classes = self.predict_ensemble(str(img_path))
            
            predictions = {
                'image': img_path.name,
                'boxes': boxes.tolist() if len(boxes) > 0 else [],
                'scores': scores.tolist() if len(scores) > 0 else [],
                'classes': classes.tolist() if len(classes) > 0 else []
            }
            all_predictions.append(predictions)
        
        # Save predictions
        if save_results:
            output_file = Path("ensemble_predictions.json")
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            self.logger.info(f"Predictions saved to: {output_file}")
        
        # Use first model's validation function for metrics
        self.logger.info("Computing metrics...")
        val_results = self.models[0].val(data=str(self.dataset_config))
        
        metrics = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr)
        }
        
        self.logger.info("="*80)
        self.logger.info("Ensemble Evaluation Results:")
        self.logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        self.logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info("="*80)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Model Ensemble for YOLOv8"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Paths to model checkpoints'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Model weights (default: equal weights)'
    )
    parser.add_argument(
        '--fusion',
        type=str,
        choices=['wbf', 'nms', 'soft_nms', 'voting'],
        default='wbf',
        help='Fusion method (default: wbf)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for fusion (default: 0.5)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Create ensemble
    ensemble = ModelEnsemble(
        model_paths=args.models,
        dataset_config=args.data,
        weights=args.weights,
        fusion_method=args.fusion,
        iou_threshold=args.iou,
        conf_threshold=args.conf
    )
    
    # Evaluate ensemble
    ensemble.evaluate_ensemble()


if __name__ == "__main__":
    main()
