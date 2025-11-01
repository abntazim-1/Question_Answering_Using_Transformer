"""
Logging utility for the project
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


class Logger:
    """Centralized logging utility"""
    
    _instance = None
    _logger = None
    
    def __new__(cls, log_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save log files
        """
        if self._logger is None:
            self._logger = self._setup_logger(log_dir)
    
    def _setup_logger(
        self,
        log_dir: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup logger with file and console handlers
        
        Args:
            log_dir: Directory to save log files
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('qa_transformer')
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_path = Path(log_dir) / 'training.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get logger instance"""
        return self._logger
    
    @classmethod
    def get_instance(cls) -> 'Logger':
        """Get singleton logger instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Convenience functions
def get_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Logger instance
    """
    logger_instance = Logger(log_dir)
    return logger_instance.get_logger()

