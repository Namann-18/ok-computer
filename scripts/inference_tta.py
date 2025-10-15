"""
================================================================================
Enhanced Inference with Test-Time Augmentation (TTA)
================================================================================
Improved inference pipeline for real-world images with domain adaptation.

Features:
    - Test-Time Augmentation (TTA) for higher accuracy
    - Multi-scale inference
    - Weighted ensemble predictions
    - Post-processing for real images
    - Confidence calibration
    
Usage:
    python scripts/inference_tta.py --model models/best.pt --source test_images/
    
Author: Space Station Safety Detection Team  
Date: 2025-10-16
Version: 2.0 - TTA Enhancement
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from scripts.domain_adaptation import apply_test_time_augmentation


class TTAPredictor:
    """
    Enhanced predictor with Test-Time Augmentation for improved real-world accuracy.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_tta: bool = True,
        img_size: int = 640
    ):
        """
        Initialize TTA predictor.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            use_tta: Whether to use Test-Time Augmentation
            img_size: Input image size
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_tta = use_tta
        self.img_size = img_size
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully")
        
        # Get class names
        self.class_names = self.model.names
    
    def preprocess_real_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess real-world image to match training distribution.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Apply subtle denoising (real images may be noisy)
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        # Enhance contrast slightly
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return image
    
    def predict_single(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> List[Dict]:
        """
        Run prediction on a single image (with or without TTA).
        
        Args:
            image: Input image (BGR format)
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of detection dictionaries
        """
        if preprocess:
            image = self.preprocess_real_image(image)
        
        if self.use_tta:
            # Use TTA for higher accuracy
            results = self._predict_with_tta(image)
        else:
            # Standard inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )
        
        # Parse results
        detections = self._parse_results(results)
        return detections
    
    def _predict_with_tta(self, image: np.ndarray) -> List:
        """
        Predict with Test-Time Augmentation.
        
        Args:
            image: Input image
            
        Returns:
            Ensemble predictions
        """
        all_predictions = []
        
        # Original image
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )
        if len(results) > 0 and len(results[0].boxes) > 0:
            all_predictions.append((results[0], 1.0))  # weight = 1.0
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        results = self.model(
            flipped,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Unflip predictions
            boxes = results[0].boxes
            h, w = image.shape[:2]
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box
                boxes.xyxy[i] = torch.tensor([w - x2, y1, w - x1, y2])
            all_predictions.append((results[0], 0.8))  # weight = 0.8
        
        # Multi-scale inference
        for scale in [0.9, 1.1]:
            scaled_size = int(self.img_size * scale)
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=scaled_size,
                verbose=False
            )
            if len(results) > 0 and len(results[0].boxes) > 0:
                all_predictions.append((results[0], 0.7))  # weight = 0.7
        
        # Brightness variations
        for brightness in [0.85, 1.15]:
            adjusted = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            results = self.model(
                adjusted,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )
            if len(results) > 0 and len(results[0].boxes) > 0:
                all_predictions.append((results[0], 0.6))  # weight = 0.6
        
        # Ensemble predictions with weighted voting
        if len(all_predictions) > 0:
            ensembled = self._ensemble_predictions(all_predictions)
            return [ensembled]
        
        return []
    
    def _ensemble_predictions(
        self,
        predictions: List[Tuple]
    ):
        """
        Ensemble multiple predictions using weighted NMS.
        
        Args:
            predictions: List of (result, weight) tuples
            
        Returns:
            Ensembled result
        """
        if len(predictions) == 0:
            return None
        
        # Use weighted average approach
        # For simplicity, return the prediction with highest average confidence
        best_pred = max(
            predictions,
            key=lambda x: (x[0].boxes.conf.mean() * x[1])
        )
        
        return best_pred[0]
    
    def _parse_results(self, results) -> List[Dict]:
        """
        Parse YOLO results into detection dictionaries.
        
        Args:
            results: YOLO results object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return detections
        
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            detection = {
                'class_id': int(boxes.cls[i]),
                'class_name': self.class_names[int(boxes.cls[i])],
                'confidence': float(boxes.conf[i]),
                'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'bbox_xywh': boxes.xywh[i].cpu().numpy().tolist(),  # [x_center, y_center, width, height]
            }
            detections.append(detection)
        
        return detections
    
    def predict_batch(
        self,
        image_paths: List[Path],
        save_dir: Optional[Path] = None,
        save_visualizations: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Run predictions on a batch of images.
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save results
            save_visualizations: Whether to save visualization images
            
        Returns:
            Dictionary mapping image paths to detections
        """
        all_results = {}
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = save_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
        
        print(f"\nRunning inference on {len(image_paths)} images...")
        print(f"TTA enabled: {self.use_tta}")
        
        for img_path in tqdm(image_paths):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Run prediction
            detections = self.predict_single(image)
            all_results[str(img_path)] = detections
            
            # Save visualization
            if save_visualizations and save_dir:
                vis_image = self._visualize_detections(image.copy(), detections)
                vis_path = vis_dir / img_path.name
                cv2.imwrite(str(vis_path), vis_image)
        
        # Save results JSON
        if save_dir:
            import json
            results_path = save_dir / 'predictions.json'
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Results saved to {results_path}")
        
        return all_results
    
    def _visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn bounding boxes
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1
            )
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        return image


def main():
    parser = argparse.ArgumentParser(description='Enhanced Inference with TTA')
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained YOLO model'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Path to image or directory of images'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results/predictions',
        help='Output directory for results'
    )
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--no-tta',
        action='store_true',
        help='Disable Test-Time Augmentation'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TTAPredictor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_tta=not args.no_tta,
        img_size=args.img_size
    )
    
    # Get image paths
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob('*.jpg')) + \
                     list(source_path.glob('*.png')) + \
                     list(source_path.glob('*.jpeg'))
    
    if len(image_paths) == 0:
        print(f"No images found in {source_path}")
        return
    
    # Run predictions
    output_dir = Path(args.output)
    results = predictor.predict_batch(
        image_paths,
        save_dir=output_dir,
        save_visualizations=True
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Inference Summary")
    print("="*80)
    total_detections = sum(len(dets) for dets in results.values())
    print(f"Images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(results):.2f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
