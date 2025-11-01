"""
Centralized configuration handling
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set defaults if not present
        self._set_defaults()
        
        return self.config
    
    def _set_defaults(self):
        """Set default values for missing configuration"""
        defaults = {
            'model': {
                'name': 'bert-base-uncased',
                'cache_dir': None
            },
            'training': {
                'num_epochs': 3,
                'batch_size': 8,
                'learning_rate': 3e-5,
                'max_length': 512
            },
            'data': {
                'train_path': 'data/raw/train.json',
                'val_path': 'data/raw/val.json',
                'processed_dir': 'data/processed'
            },
            'output': {
                'model_dir': 'outputs/models',
                'results_dir': 'outputs/results',
                'log_dir': 'outputs/logs'
            }
        }
        
        # Merge defaults with loaded config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save configuration (uses config_path if None)
        """
        path = output_path or self.config_path
        if not path:
            raise ValueError("No path provided to save configuration")
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment"""
        self.set(key, value)


def load_config(config_path: str) -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)

