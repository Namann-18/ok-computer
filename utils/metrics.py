"""
================================================================================
Metrics Calculation Module
================================================================================
Production-grade metrics calculation for object detection evaluation.
Implements standard detection metrics (mAP, precision, recall, F1) and
scenario-specific analysis for space station safety object detection.

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class DetectionMetrics:
    """
    Container for object detection metrics.
    
    Attributes:
        map50: Mean Average Precision at IoU=0.5
        map50_95: Mean Average Precision at IoU=0.5:0.95
        precision: Overall precision
        recall: Overall recall
        f1_score: F1 score (harmonic mean of precision and recall)
        per_class_ap: Average Precision per class
        per_class_precision: Precision per class
        per_class_recall: Recall per class
        confusion_matrix: Confusion matrix (if computed)
    """
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1_score: float
    per_class_ap: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        metrics_dict = asdict(self)
        if self.confusion_matrix is not None:
            metrics_dict['confusion_matrix'] = self.confusion_matrix.tolist()
        return metrics_dict
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'DetectionMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'confusion_matrix' in data and data['confusion_matrix'] is not None:
            data['confusion_matrix'] = np.array(data['confusion_matrix'])
        
        return cls(**data)


class MetricsCalculator:
    """
    Calculate comprehensive object detection metrics.
    
    Provides methods for computing standard detection metrics and
    scenario-specific analysis for space station safety detection.
    
    Example:
        >>> calculator = MetricsCalculator(class_names=['OxygenTank', 'FireAlarm'])
        >>> metrics = calculator.compute_metrics(predictions, ground_truth)
        >>> print(f"mAP@0.5: {metrics.map50:.3f}")
    """
    
    def __init__(
        self,
        class_names: List[str],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.001
    ) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names in order.
            iou_threshold: IoU threshold for considering a detection as TP.
            conf_threshold: Confidence threshold for filtering predictions.
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
    
    @staticmethod
    def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: Bounding box [x1, y1, x2, y2].
            box2: Bounding box [x1, y1, x2, y2].
        
        Returns:
            IoU value between 0 and 1.
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    @staticmethod
    def box_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between multiple boxes (vectorized).
        
        Args:
            boxes1: Array of boxes [N, 4] in format [x1, y1, x2, y2].
            boxes2: Array of boxes [M, 4] in format [x1, y1, x2, y2].
        
        Returns:
            IoU matrix [N, M].
        """
        # Calculate intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = boxes1_area[:, None] + boxes2_area - intersection
        
        # Calculate IoU
        iou = intersection / np.maximum(union, 1e-10)
        return iou
    
    def calculate_ap(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
        method: str = "interp"
    ) -> float:
        """
        Calculate Average Precision (AP) from precision-recall curve.
        
        Args:
            precisions: Array of precision values.
            recalls: Array of recall values.
            method: Calculation method ('interp' for 11-point interpolation,
                   'continuous' for area under curve).
        
        Returns:
            Average Precision value.
        """
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        precisions = precisions[sorted_indices]
        recalls = recalls[sorted_indices]
        
        if method == "interp":
            # 11-point interpolation (PASCAL VOC style)
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0
        else:
            # Continuous (COCO style)
            # Add sentinel values
            precisions = np.concatenate(([0], precisions, [0]))
            recalls = np.concatenate(([0], recalls, [1]))
            
            # Ensure precision is monotonically decreasing
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            
            # Calculate area under curve
            indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
            ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        
        return ap
    
    def compute_confusion_matrix(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> np.ndarray:
        """
        Compute confusion matrix for object detection.
        
        Args:
            predictions: List of prediction dictionaries with keys:
                        'boxes' (N, 4), 'scores' (N,), 'labels' (N,)
            ground_truths: List of ground truth dictionaries with keys:
                          'boxes' (M, 4), 'labels' (M,)
        
        Returns:
            Confusion matrix of shape (num_classes + 1, num_classes + 1)
            where last row/col represents background/false detections.
        """
        # Initialize confusion matrix (add 1 for background class)
        cm = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        
        # Process each image
        for pred, gt in zip(predictions, ground_truths):
            if len(gt['labels']) == 0:
                # No ground truth objects
                if len(pred['labels']) > 0:
                    # False positives
                    for pred_label in pred['labels']:
                        cm[pred_label, self.num_classes] += 1
                continue
            
            if len(pred['labels']) == 0:
                # No predictions, all ground truth are false negatives
                for gt_label in gt['labels']:
                    cm[self.num_classes, gt_label] += 1
                continue
            
            # Calculate IoU between all predictions and ground truths
            iou_matrix = self.box_iou_batch(pred['boxes'], gt['boxes'])
            
            # Match predictions to ground truths
            matched_gt = set()
            
            for pred_idx in range(len(pred['labels'])):
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt['labels'])):
                    if gt_idx in matched_gt:
                        continue
                    
                    if iou_matrix[pred_idx, gt_idx] > best_iou:
                        best_iou = iou_matrix[pred_idx, gt_idx]
                        best_gt_idx = gt_idx
                
                pred_label = pred['labels'][pred_idx]
                
                if best_iou >= self.iou_threshold and best_gt_idx != -1:
                    # Correct localization
                    gt_label = gt['labels'][best_gt_idx]
                    cm[pred_label, gt_label] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # False positive (background)
                    cm[pred_label, self.num_classes] += 1
            
            # Add false negatives (unmatched ground truths)
            for gt_idx in range(len(gt['labels'])):
                if gt_idx not in matched_gt:
                    gt_label = gt['labels'][gt_idx]
                    cm[self.num_classes, gt_label] += 1
        
        return cm
    
    def format_metrics_table(self, metrics: DetectionMetrics) -> str:
        """
        Format metrics as a readable table string.
        
        Args:
            metrics: DetectionMetrics object.
        
        Returns:
            Formatted string representation.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DETECTION METRICS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"{'mAP@0.5:':<30} {metrics.map50:>10.4f}")
        lines.append(f"{'mAP@0.5:0.95:':<30} {metrics.map50_95:>10.4f}")
        lines.append(f"{'Precision:':<30} {metrics.precision:>10.4f}")
        lines.append(f"{'Recall:':<30} {metrics.recall:>10.4f}")
        lines.append(f"{'F1-Score:':<30} {metrics.f1_score:>10.4f}")
        lines.append("-" * 80)
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 80)
        lines.append(f"{'Class':<20} {'AP@0.5':>12} {'Precision':>12} {'Recall':>12}")
        lines.append("-" * 80)
        
        for class_name in self.class_names:
            ap = metrics.per_class_ap.get(class_name, 0.0)
            prec = metrics.per_class_precision.get(class_name, 0.0)
            rec = metrics.per_class_recall.get(class_name, 0.0)
            lines.append(f"{class_name:<20} {ap:>12.4f} {prec:>12.4f} {rec:>12.4f}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics calculator
    class_names = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm',
                   'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
    
    calculator = MetricsCalculator(class_names=class_names)
    
    # Test IoU calculation
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([20, 20, 60, 60])
    iou = calculator.box_iou(box1, box2)
    print(f"IoU between box1 and box2: {iou:.4f}")
    
    # Test metrics object
    metrics = DetectionMetrics(
        map50=0.8534,
        map50_95=0.7234,
        precision=0.8712,
        recall=0.8234,
        f1_score=0.8465,
        per_class_ap={name: 0.85 for name in class_names},
        per_class_precision={name: 0.87 for name in class_names},
        per_class_recall={name: 0.82 for name in class_names}
    )
    
    print(calculator.format_metrics_table(metrics))
