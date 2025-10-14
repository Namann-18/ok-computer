"""
================================================================================
Logger Utility Module
================================================================================
Production-grade logging system for training, evaluation, and inference.
Provides structured logging with multiple output streams and formatting options.

Author: Space Station Safety Detection Team
Date: 2025-10-14
Version: 1.0
================================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import colorlog


class Logger:
    """
    Production-grade logger with colored console output and file logging.
    
    Features:
        - Colored console output for better readability
        - File logging with rotation
        - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - Structured log formatting
        - Thread-safe logging
    
    Attributes:
        logger (logging.Logger): The underlying Python logger instance.
        log_file (Path): Path to the log file.
    
    Example:
        >>> logger = Logger(name="training", log_dir="./logs")
        >>> logger.info("Training started")
        >>> logger.warning("Learning rate decreased")
        >>> logger.error("Training failed", exc_info=True)
    """
    
    def __init__(
        self,
        name: str = "yolov8m",
        log_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True
    ) -> None:
        """
        Initialize logger with console and file handlers.
        
        Args:
            name: Logger name (usually module or script name).
            log_dir: Directory to store log files.
            log_file: Custom log filename (default: {name}_{timestamp}.log).
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            console: Enable console logging.
            file: Enable file logging.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Remove existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Setup console handler with colors
        if console:
            self._setup_console_handler()
        
        # Setup file handler
        if file:
            self._setup_file_handler(log_dir, log_file, name)
    
    def _setup_console_handler(self) -> None:
        """Setup colored console output handler."""
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Colored formatter
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(
        self,
        log_dir: Optional[Union[str, Path]],
        log_file: Optional[str],
        name: str
    ) -> None:
        """Setup file output handler."""
        # Create log directory
        if log_dir is None:
            log_dir = Path("./logs")
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        self.log_file = log_dir / log_file
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # File formatter (without colors)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log DEBUG level message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log INFO level message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log WARNING level message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log ERROR level message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log CRITICAL level message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """
        Log metrics in a structured format.
        
        Args:
            metrics: Dictionary of metric name: value pairs.
            prefix: Optional prefix for metric names.
        
        Example:
            >>> logger.log_metrics({"mAP": 0.85, "precision": 0.87}, prefix="val")
        """
        metrics_str = " | ".join([f"{prefix}/{k if prefix else k}: {v:.4f}" 
                                  for k, v in metrics.items()])
        self.info(f"Metrics: {metrics_str}")
    
    def log_config(self, config: dict, title: str = "Configuration") -> None:
        """
        Log configuration dictionary in readable format.
        
        Args:
            config: Configuration dictionary.
            title: Title for the configuration section.
        """
        self.info(f"{'='*80}")
        self.info(f"{title}")
        self.info(f"{'='*80}")
        
        def log_dict(d: dict, indent: int = 0):
            """Recursively log nested dictionaries."""
            for key, value in d.items():
                if isinstance(value, dict):
                    self.info(f"{'  ' * indent}{key}:")
                    log_dict(value, indent + 1)
                else:
                    self.info(f"{'  ' * indent}{key}: {value}")
        
        log_dict(config)
        self.info(f"{'='*80}")


def get_logger(
    name: str = "yolov8m",
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> Logger:
    """
    Factory function to create a logger instance.
    
    Args:
        name: Logger name.
        log_dir: Directory to store log files.
        level: Logging level.
    
    Returns:
        Configured Logger instance.
    
    Example:
        >>> logger = get_logger("train", "./logs", logging.DEBUG)
        >>> logger.info("Model training started")
    """
    return Logger(name=name, log_dir=log_dir, level=level)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test", "./logs")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test metrics logging
    metrics = {
        "mAP@0.5": 0.8534,
        "precision": 0.8712,
        "recall": 0.8234,
        "F1": 0.8465
    }
    logger.log_metrics(metrics, prefix="val")
    
    # Test config logging
    config = {
        "model": {"architecture": "yolov8m", "pretrained": True},
        "training": {"epochs": 300, "batch_size": 16, "lr": 0.001}
    }
    logger.log_config(config, "Training Configuration")
