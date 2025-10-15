"""
================================================================================
Advanced Data Augmentation for YOLOv8
================================================================================
Generate synthetic training samples with advanced augmentation techniques
to improve model robustness and generalization.

Techniques:
    - MixUp: Blend images and labels
    - CutMix: Cut and paste patches
    - Mosaic: Combine 4 images into one
    - Random erasing: Simulate occlusions
    - Color jittering: Lighting variations
    - Noise injection: Simulate sensor noise
    - Geometric transformations: Rotation, scaling, perspective
    
Goal: Expand dataset strategically to improve mAP from 0.846 to 0.90+

Author: Space Station Safety Detection Team
Date: 2025-10-15
Version: 1.0
================================================================================
"""

import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger


class AdvancedAugmentor:
    """
    Advanced data augmentation pipeline for object detection.
    """
    
    def __init__(
        self,
        dataset_config: str,
        output_dir: str,
        augmentation_factor: int = 3
    ):
        """
        Initialize augmentor.
        
        Args:
            dataset_config: Path to dataset configuration
            output_dir: Directory for augmented dataset
            augmentation_factor: Multiply dataset by this factor
        """
        self.dataset_config = Path(dataset_config)
        self.output_dir = Path(output_dir)
        self.augmentation_factor = augmentation_factor
        
        # Load dataset config
        with open(self.dataset_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = get_logger(
            name="advanced_augmentation",
            log_dir=str(self.output_dir / "logs")
        )
        
        # Define augmentation pipelines
        self.setup_augmentation_pipelines()
        
        self.logger.info("="*80)
        self.logger.info("Advanced Augmentation Pipeline Initialized")
        self.logger.info(f"Augmentation Factor: {augmentation_factor}x")
        self.logger.info("="*80)
    
    def setup_augmentation_pipelines(self):
        """Setup multiple augmentation pipelines for diversity."""
        
        # Pipeline 1: Lighting and color variations
        self.pipeline_lighting = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline 2: Geometric transformations
        self.pipeline_geometric = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline 3: Noise and quality degradation
        self.pipeline_noise = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.4),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline 4: Occlusion and cutout
        self.pipeline_occlusion = A.Compose([
            A.CoarseDropout(
                max_holes=8,
                max_height=50,
                max_width=50,
                min_holes=1,
                min_height=20,
                min_width=20,
                fill_value=0,
                p=0.5
            ),
            A.GridDropout(ratio=0.3, p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline 5: Extreme conditions
        self.pipeline_extreme = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.5,
                contrast_limit=0.5,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(50, 150), p=0.7),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        self.pipelines = [
            ('lighting', self.pipeline_lighting),
            ('geometric', self.pipeline_geometric),
            ('noise', self.pipeline_noise),
            ('occlusion', self.pipeline_occlusion),
            ('extreme', self.pipeline_extreme),
        ]
    
    def load_yolo_annotations(self, label_path: Path) -> Tuple[List[int], List[List[float]]]:
        """
        Load YOLO format annotations.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (class_labels, bboxes)
        """
        class_labels = []
        bboxes = []
        
        if not label_path.exists():
            return class_labels, bboxes
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    class_labels.append(cls)
                    bboxes.append([x, y, w, h])
        
        return class_labels, bboxes
    
    def save_yolo_annotations(
        self,
        label_path: Path,
        class_labels: List[int],
        bboxes: List[List[float]]
    ):
        """
        Save YOLO format annotations.
        
        Args:
            label_path: Output label file path
            class_labels: List of class IDs
            bboxes: List of bounding boxes in YOLO format
        """
        with open(label_path, 'w') as f:
            for cls, bbox in zip(class_labels, bboxes):
                x, y, w, h = bbox
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    def augment_single_image(
        self,
        image: np.ndarray,
        class_labels: List[int],
        bboxes: List[List[float]],
        pipeline_name: str
    ) -> Tuple[np.ndarray, List[int], List[List[float]]]:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image
            class_labels: Class labels
            bboxes: Bounding boxes in YOLO format
            pipeline_name: Name of augmentation pipeline to use
            
        Returns:
            Augmented image, class_labels, and bboxes
        """
        # Select pipeline
        pipeline = dict(self.pipelines)[pipeline_name]
        
        # Apply augmentation
        try:
            transformed = pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return (
                transformed['image'],
                transformed['class_labels'],
                transformed['bboxes']
            )
        except Exception as e:
            self.logger.warning(f"Augmentation failed: {e}")
            return image, class_labels, bboxes
    
    def create_mosaic(
        self,
        images: List[np.ndarray],
        labels_list: List[Tuple[List[int], List[List[float]]]],
        output_size: int = 640
    ) -> Tuple[np.ndarray, List[int], List[List[float]]]:
        """
        Create mosaic augmentation from 4 images.
        
        Args:
            images: List of 4 images
            labels_list: List of 4 (class_labels, bboxes) tuples
            output_size: Output image size
            
        Returns:
            Mosaic image, combined class_labels, and adjusted bboxes
        """
        # Create mosaic canvas
        mosaic = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # Random center point
        center_x = random.randint(output_size // 4, 3 * output_size // 4)
        center_y = random.randint(output_size // 4, 3 * output_size // 4)
        
        all_class_labels = []
        all_bboxes = []
        
        # Place 4 images
        positions = [
            (0, 0, center_x, center_y),           # Top-left
            (center_x, 0, output_size, center_y), # Top-right
            (0, center_y, center_x, output_size), # Bottom-left
            (center_x, center_y, output_size, output_size) # Bottom-right
        ]
        
        for i, (img, (class_labels, bboxes)) in enumerate(zip(images[:4], labels_list[:4])):
            x1, y1, x2, y2 = positions[i]
            w, h = x2 - x1, y2 - y1
            
            # Resize image to fit
            img_resized = cv2.resize(img, (w, h))
            mosaic[y1:y2, x1:x2] = img_resized
            
            # Adjust bboxes
            for cls, bbox in zip(class_labels, bboxes):
                # Convert from YOLO to absolute coordinates
                img_h, img_w = img.shape[:2]
                x_center = bbox[0] * img_w
                y_center = bbox[1] * img_h
                bbox_w = bbox[2] * img_w
                bbox_h = bbox[3] * img_h
                
                # Scale to mosaic section
                new_x_center = x1 + (x_center / img_w) * w
                new_y_center = y1 + (y_center / img_h) * h
                new_bbox_w = (bbox_w / img_w) * w
                new_bbox_h = (bbox_h / img_h) * h
                
                # Convert back to YOLO format
                new_bbox = [
                    new_x_center / output_size,
                    new_y_center / output_size,
                    new_bbox_w / output_size,
                    new_bbox_h / output_size
                ]
                
                # Check if bbox is valid
                if (new_bbox[2] > 0.01 and new_bbox[3] > 0.01 and
                    new_bbox[0] > 0 and new_bbox[0] < 1 and
                    new_bbox[1] > 0 and new_bbox[1] < 1):
                    all_class_labels.append(cls)
                    all_bboxes.append(new_bbox)
        
        return mosaic, all_class_labels, all_bboxes
    
    def augment_dataset(self):
        """Main augmentation workflow."""
        self.logger.info("Starting dataset augmentation...")
        
        # Setup directories
        dataset_root = Path(self.data_config['path'])
        train_images = dataset_root / self.data_config['train']
        train_labels = dataset_root / "train" / "labels"
        
        output_images = self.output_dir / "images"
        output_labels = self.output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Get all training images
        image_files = list(train_images.glob("*.jpg")) + \
                     list(train_images.glob("*.png")) + \
                     list(train_images.glob("*.jpeg"))
        
        self.logger.info(f"Found {len(image_files)} training images")
        
        total_generated = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc="Augmenting images"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Load labels
            label_path = train_labels / f"{img_path.stem}.txt"
            class_labels, bboxes = self.load_yolo_annotations(label_path)
            
            if not bboxes:
                continue  # Skip images without annotations
            
            # Generate augmented versions
            for aug_idx in range(self.augmentation_factor):
                # Randomly select augmentation pipeline
                pipeline_name, _ = random.choice(self.pipelines)
                
                # Apply augmentation
                aug_image, aug_labels, aug_bboxes = self.augment_single_image(
                    image.copy(),
                    class_labels.copy(),
                    bboxes.copy(),
                    pipeline_name
                )
                
                if not aug_bboxes:
                    continue
                
                # Save augmented image
                output_name = f"{img_path.stem}_aug{aug_idx}_{pipeline_name}{img_path.suffix}"
                output_img_path = output_images / output_name
                cv2.imwrite(str(output_img_path), aug_image)
                
                # Save augmented labels
                output_label_path = output_labels / f"{output_img_path.stem}.txt"
                self.save_yolo_annotations(
                    output_label_path,
                    aug_labels,
                    aug_bboxes
                )
                
                total_generated += 1
        
        # Generate mosaic augmentations
        self.logger.info("Generating mosaic augmentations...")
        num_mosaics = len(image_files) // 4
        
        for mosaic_idx in tqdm(range(num_mosaics), desc="Creating mosaics"):
            # Randomly select 4 images
            selected_imgs = random.sample(image_files, 4)
            images = []
            labels_list = []
            
            for img_path in selected_imgs:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                images.append(img)
                
                label_path = train_labels / f"{img_path.stem}.txt"
                labels_list.append(self.load_yolo_annotations(label_path))
            
            if len(images) == 4:
                # Create mosaic
                mosaic_img, mosaic_labels, mosaic_bboxes = self.create_mosaic(
                    images, labels_list
                )
                
                if mosaic_bboxes:
                    # Save mosaic
                    mosaic_name = f"mosaic_{mosaic_idx}.jpg"
                    mosaic_path = output_images / mosaic_name
                    cv2.imwrite(str(mosaic_path), mosaic_img)
                    
                    # Save labels
                    mosaic_label_path = output_labels / f"mosaic_{mosaic_idx}.txt"
                    self.save_yolo_annotations(
                        mosaic_label_path,
                        mosaic_labels,
                        mosaic_bboxes
                    )
                    
                    total_generated += 1
        
        self.logger.info(f"✓ Generated {total_generated} augmented images")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Print class distribution
        self.print_class_distribution(output_labels)
    
    def print_class_distribution(self, labels_dir: Path):
        """Print class distribution in augmented dataset."""
        class_counts = {i: 0 for i in range(self.data_config['nc'])}
        
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        class_counts[cls] += 1
        
        self.logger.info("\nClass distribution in augmented dataset:")
        total = sum(class_counts.values())
        for cls_id, count in class_counts.items():
            cls_name = self.data_config['names'][cls_id]
            percentage = (count / total) * 100 if total > 0 else 0
            self.logger.info(f"  {cls_name}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Data Augmentation for YOLOv8"
    )
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./augmented_dataset',
        help='Output directory for augmented dataset'
    )
    parser.add_argument(
        '--factor',
        type=int,
        default=3,
        help='Augmentation factor (multiply dataset size)'
    )
    
    args = parser.parse_args()
    
    # Create augmentor
    augmentor = AdvancedAugmentor(
        dataset_config=args.data,
        output_dir=args.output,
        augmentation_factor=args.factor
    )
    
    # Run augmentation
    augmentor.augment_dataset()


if __name__ == "__main__":
    main()
