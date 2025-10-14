"""
================================================================================
Utilities Package
================================================================================
Production-grade utility modules for space station safety object detection.

Modules:
    - logger: Structured logging with console and file output
    - metrics: Comprehensive object detection metrics calculation
    - visualization: Drawing bounding boxes and creating evaluation plots
    - callbacks: Training callbacks (early stopping, checkpointing, etc.)

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

from .logger import Logger, get_logger
from .metrics import MetricsCalculator, DetectionMetrics
from .visualization import (
    draw_bounding_boxes,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_training_curves,
    create_detection_grid
)
from .callbacks import EarlyStopping, ModelCheckpoint, MetricsTracker

__all__ = [
    # Logger
    'Logger',
    'get_logger',
    
    # Metrics
    'MetricsCalculator',
    'DetectionMetrics',
    
    # Visualization
    'draw_bounding_boxes',
    'plot_confusion_matrix',
    'plot_precision_recall_curve',
    'plot_training_curves',
    'create_detection_grid',
    
    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
    'MetricsTracker',
]

__version__ = '1.0.0'
__author__ = 'Space Station Safety Detection Team'
