"""
Logging utilities for structured logging across the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logger with file and console handlers.
    
    Args:
        name: Logger name (default: root logger)
        log_dir: Directory to save log files
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class MetricsLogger:
    """
    Logger for tracking metrics during training.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Optional file to save metrics
        """
        self.metrics_history = []
        self.log_file = log_file
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, epoch: int, metrics: dict):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {'epoch': epoch, **metrics}
        self.metrics_history.append(entry)
        
        if self.log_file:
            self._write_to_file(entry)
    
    def _write_to_file(self, entry: dict):
        """Write metrics entry to file."""
        import json
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_history(self) -> list:
        """Get complete metrics history."""
        return self.metrics_history


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(
        name="test_logger",
        log_dir=Path("logs"),
        level="INFO"
    )
    
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
