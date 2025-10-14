"""
================================================================================
Visualization Utility Module
================================================================================
Production-grade visualization tools for object detection results.
Provides functions for drawing bounding boxes, creating evaluation plots,
and generating comprehensive visual reports.

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# Color palette for different classes (distinct colors for visibility)
CLASS_COLORS = [
    (255, 87, 51),    # OxygenTank - Red-Orange
    (51, 153, 255),   # NitrogenTank - Sky Blue
    (255, 195, 0),    # FirstAidBox - Gold
    (255, 51, 51),    # FireAlarm - Red
    (102, 255, 102),  # SafetySwitchPanel - Light Green
    (204, 102, 255),  # EmergencyPhone - Purple
    (255, 128, 0),    # FireExtinguisher - Orange
]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on image with labels and confidence scores.
    
    Args:
        image: Input image (H, W, 3) in BGR format.
        boxes: Bounding boxes (N, 4) in format [x1, y1, x2, y2].
        labels: Class labels (N,) as integers.
        scores: Confidence scores (N,) between 0 and 1.
        class_names: List of class names for label display.
        colors: List of RGB colors for each class.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for text.
        show_confidence: Whether to display confidence scores.
    
    Returns:
        Image with bounding boxes drawn.
    """
    # Make a copy to avoid modifying original
    image = image.copy()
    
    # Use default colors if not provided
    if colors is None:
        colors = CLASS_COLORS
    
    # Ensure colors list is long enough
    while len(colors) < len(np.unique(labels)):
        colors.extend(CLASS_COLORS)
    
    # Draw each bounding box
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if class_names is not None:
            label_text = class_names[int(label)]
        else:
            label_text = f"Class {label}"
        
        if scores is not None and show_confidence:
            label_text += f" {scores[i]:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness=1,
            lineType=cv2.LINE_AA
        )
    
    return image


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix (N, N) or (N+1, N+1) with background.
        class_names: List of class names.
        normalize: Whether to normalize by row (true labels).
        figsize: Figure size (width, height).
        cmap: Colormap name.
        save_path: Path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add background class if needed
    if confusion_matrix.shape[0] == len(class_names) + 1:
        class_names = class_names + ['Background']
    
    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    else:
        cm = confusion_matrix
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    class_name: str = "All Classes",
    ap: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values.
        recall: Recall values.
        class_name: Name of the class.
        ap: Average Precision value to display.
        figsize: Figure size.
        save_path: Path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    label = f"{class_name}"
    if ap is not None:
        label += f" (AP = {ap:.3f})"
    
    ax.plot(recall, precision, linewidth=2, label=label)
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot training and validation curves (loss, mAP, etc.).
    
    Args:
        metrics_history: Dictionary with keys like 'train_loss', 'val_loss',
                        'val_map50', etc. Values are lists of metrics per epoch.
        figsize: Figure size.
        save_path: Path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(metrics_history.get('train_loss', [])) + 1)
    
    # Plot 1: Loss
    if 'train_loss' in metrics_history:
        axes[0].plot(epochs, metrics_history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in metrics_history:
        axes[0].plot(epochs, metrics_history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: mAP
    if 'val_map50' in metrics_history:
        axes[1].plot(epochs, metrics_history['val_map50'], label='mAP@0.5', 
                    linewidth=2, color='green')
    if 'val_map50_95' in metrics_history:
        axes[1].plot(epochs, metrics_history['val_map50_95'], label='mAP@0.5:0.95', 
                    linewidth=2, color='blue')
    axes[1].axhline(y=0.80, color='r', linestyle='--', label='Target (80%)')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('mAP', fontweight='bold')
    axes[1].set_title('Mean Average Precision', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    if 'val_precision' in metrics_history:
        axes[2].plot(epochs, metrics_history['val_precision'], label='Precision', 
                    linewidth=2, color='orange')
    if 'val_recall' in metrics_history:
        axes[2].plot(epochs, metrics_history['val_recall'], label='Recall', 
                    linewidth=2, color='purple')
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Score', fontweight='bold')
    axes[2].set_title('Precision and Recall', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_detection_grid(
    images: List[np.ndarray],
    predictions: List[Dict],
    ground_truths: Optional[List[Dict]] = None,
    class_names: Optional[List[str]] = None,
    grid_size: Tuple[int, int] = (2, 4),
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a grid of images with predictions and ground truth boxes.
    
    Args:
        images: List of images.
        predictions: List of prediction dictionaries.
        ground_truths: List of ground truth dictionaries (optional).
        class_names: List of class names.
        grid_size: Grid layout (rows, cols).
        figsize: Figure size.
        save_path: Path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (img, pred) in enumerate(zip(images[:rows * cols], predictions[:rows * cols])):
        # Convert image for display
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        
        # Draw predictions
        if len(pred['boxes']) > 0:
            img_display = draw_bounding_boxes(
                cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR),
                pred['boxes'],
                pred['labels'],
                pred.get('scores'),
                class_names,
                show_confidence=True
            )
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        title = f"Image {idx + 1}"
        if len(pred['boxes']) > 0:
            title += f" ({len(pred['boxes'])} objects)"
        axes[idx].set_title(title, fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(images), rows * cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)  # Gray background
    
    # Dummy boxes and labels
    boxes = np.array([
        [100, 100, 200, 200],
        [300, 300, 450, 450],
        [150, 400, 300, 550]
    ])
    labels = np.array([0, 2, 4])
    scores = np.array([0.95, 0.87, 0.92])
    class_names = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm',
                   'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
    
    # Test bounding box drawing
    img_with_boxes = draw_bounding_boxes(img, boxes, labels, scores, class_names)
    print("✓ Bounding box drawing test passed")
    
    # Test confusion matrix plot
    cm = np.random.randint(0, 100, (7, 7))
    fig = plot_confusion_matrix(cm, class_names, normalize=True)
    plt.close(fig)
    print("✓ Confusion matrix plot test passed")
    
    print("All visualization tests passed!")
