"""
================================================================================
Domain Adaptation Module for Synthetic-to-Real Transfer
================================================================================
Specialized augmentation techniques to bridge the gap between synthetic (Falcon)
training data and real-world images.

Key Techniques:
    1. Real-world image artifacts (blur, compression, noise)
    2. Camera sensor simulation (ISO noise, motion blur)
    3. Environmental effects (fog, dust, reflections)
    4. Color distribution matching
    5. Texture randomization
    
Goal: Improve real-world inference accuracy by making synthetic data more realistic

Author: Space Station Safety Detection Team
Date: 2025-10-16
Version: 2.0 - Domain Adaptation Enhancement
================================================================================
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DomainAdaptationAugmentor:
    """
    Advanced augmentation pipeline specifically designed to reduce domain gap
    between synthetic training data and real-world images.
    """
    
    def __init__(self, probability: float = 0.7):
        """
        Initialize domain adaptation augmentor.
        
        Args:
            probability: Probability of applying domain adaptation augmentations
        """
        self.probability = probability
        self.setup_pipelines()
    
    def setup_pipelines(self):
        """Setup multiple augmentation pipelines for domain adaptation."""
        
        # Pipeline 1: Real-world image quality degradation
        self.real_world_artifacts = A.Compose([
            # Simulate camera sensor noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.6),
            
            # Simulate motion blur (camera/object movement)
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            
            # Simulate JPEG compression artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            
            # Simulate lens distortion
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            
            # Simulate chromatic aberration
            A.ToGray(p=0.05),  # Occasional grayscale
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Pipeline 2: Environmental and lighting conditions
        self.environmental_effects = A.Compose([
            # Simulate various lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=0.8
            ),
            
            # Simulate color temperature variations (warm/cool lighting)
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            
            # Simulate shadows and uneven lighting
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.4
            ),
            
            # Simulate fog/haze (atmospheric effects)
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.08,
                p=0.2
            ),
            
            # Simulate exposure problems
            A.OneOf([
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.5),
            
            # Simulate color shift (different cameras/sensors)
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
            
            # Simulate vignetting (lens artifacts)
            A.RandomToneCurve(scale=0.1, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Pipeline 3: Advanced realistic augmentation
        self.advanced_realism = A.Compose([
            # Simulate sun glare / bright spots
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=2,
                src_radius=100,
                p=0.15
            ),
            
            # Simulate rain/water droplets
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.7,
                rain_type='drizzle',
                p=0.1
            ),
            
            # Simulate dust/particles on lens
            A.Spatter(
                mean=0.2,
                std=0.3,
                gauss_sigma=2,
                cutout_threshold=0.5,
                intensity=0.6,
                mode='rain',
                p=0.2
            ),
            
            # Add subtle channel shuffle for color variation
            A.ChannelShuffle(p=0.05),
            
            # Simulate downsampling and upsampling (different resolutions)
            A.Downscale(
                scale_min=0.7,
                scale_max=0.9,
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Pipeline 4: Texture and detail modification
        self.texture_modification = A.Compose([
            # Adjust sharpness
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                A.UnsharpMask(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            
            # Simulate different bit depths
            A.Posterize(num_bits=4, p=0.1),
            
            # Simulate quantization
            A.Equalize(mode='cv', p=0.2),
            
            # Simulate solarization (overexposure)
            A.Solarize(threshold=128, p=0.1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def apply(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply domain adaptation augmentations to an image.
        
        Args:
            image: Input image (numpy array)
            bboxes: Bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: Class labels for each bounding box
            
        Returns:
            Augmented image, bboxes, and class_labels
        """
        if random.random() > self.probability:
            return image, bboxes, class_labels
        
        # Apply pipelines sequentially with decreasing probability
        pipelines = [
            (self.real_world_artifacts, 0.9),
            (self.environmental_effects, 0.8),
            (self.advanced_realism, 0.5),
            (self.texture_modification, 0.4),
        ]
        
        for pipeline, prob in pipelines:
            if random.random() < prob:
                try:
                    augmented = pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    image = augmented['image']
                    bboxes = augmented['bboxes']
                    class_labels = augmented['class_labels']
                except Exception as e:
                    # Skip if augmentation fails
                    continue
        
        return image, bboxes, class_labels


def get_tta_transforms(img_size: int = 640) -> List[A.Compose]:
    """
    Get Test-Time Augmentation transforms for inference.
    
    TTA applies multiple augmentations during inference and ensembles
    the results for higher accuracy.
    
    Args:
        img_size: Input image size
        
    Returns:
        List of augmentation pipelines for TTA
    """
    transforms = []
    
    # Original image (no augmentation)
    transforms.append(A.Compose([
        A.Resize(img_size, img_size),
    ]))
    
    # Horizontal flip
    transforms.append(A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
    ]))
    
    # Brightness variations
    for brightness in [0.9, 1.1]:
        transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(
                brightness_limit=(brightness - 1.0, brightness - 1.0),
                contrast_limit=0,
                p=1.0
            ),
        ]))
    
    # Scale variations
    for scale in [0.9, 1.1]:
        transforms.append(A.Compose([
            A.Resize(int(img_size * scale), int(img_size * scale)),
            A.Resize(img_size, img_size),
        ]))
    
    return transforms


def apply_test_time_augmentation(
    model,
    image: np.ndarray,
    img_size: int = 640,
    conf_threshold: float = 0.25
) -> List:
    """
    Apply Test-Time Augmentation to improve inference accuracy.
    
    Args:
        model: YOLO model instance
        image: Input image
        img_size: Model input size
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Ensemble predictions
    """
    tta_transforms = get_tta_transforms(img_size)
    all_predictions = []
    
    # Run inference with each augmentation
    for transform in tta_transforms:
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # Run inference
        results = model(aug_image, conf=conf_threshold, verbose=False)
        
        # Collect predictions
        if len(results) > 0 and len(results[0].boxes) > 0:
            all_predictions.append(results[0])
    
    # Ensemble predictions (weighted voting / NMS)
    if len(all_predictions) > 0:
        # Simple approach: return predictions with highest confidence
        return max(all_predictions, key=lambda x: x.boxes.conf.mean())
    
    return []


def create_domain_adapted_dataset(
    source_dir: str,
    output_dir: str,
    augmentation_factor: int = 2
):
    """
    Create domain-adapted version of synthetic dataset.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for adapted dataset
        augmentation_factor: Number of augmented versions per image
    """
    from pathlib import Path
    import shutil
    from tqdm import tqdm
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    augmentor = DomainAdaptationAugmentor(probability=1.0)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_img_dir = source_path / split / 'images'
        split_label_dir = source_path / split / 'labels'
        
        if not split_img_dir.exists():
            continue
        
        output_img_dir = output_path / split / 'images'
        output_label_dir = output_path / split / 'labels'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(split_img_dir.glob('*.jpg')) + list(split_img_dir.glob('*.png'))
        
        print(f"\nProcessing {split} split ({len(image_files)} images)...")
        
        for img_file in tqdm(image_files):
            # Copy original
            shutil.copy(img_file, output_img_dir / img_file.name)
            
            label_file = split_label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, output_label_dir / label_file.name)
            
            # Create augmented versions
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load labels
            bboxes = []
            class_labels = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            class_labels.append(cls)
                            bboxes.append(bbox)
            
            # Generate augmented versions
            for i in range(augmentation_factor):
                aug_image, aug_bboxes, aug_labels = augmentor.apply(
                    image.copy(), bboxes.copy(), class_labels.copy()
                )
                
                # Save augmented image
                aug_img_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                aug_img_path = output_img_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # Save augmented labels
                aug_label_name = f"{img_file.stem}_aug{i}.txt"
                aug_label_path = output_label_dir / aug_label_name
                with open(aug_label_path, 'w') as f:
                    for cls, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    print(f"\nDomain-adapted dataset created at: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Adaptation for Synthetic-to-Real Transfer")
    parser.add_argument('--source', type=str, required=True, help='Source dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--aug-factor', type=int, default=2, help='Augmentation factor')
    
    args = parser.parse_args()
    
    create_domain_adapted_dataset(args.source, args.output, args.aug_factor)
