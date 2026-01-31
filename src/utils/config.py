"""
Configuration loader for TCGA Multi-modal Classification.
Centralizes all hyperparameters - NO MAGIC NUMBERS in code!
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


class Config:
    """
    Configuration class with attribute access.
    
    Usage:
        config = Config("configs/config.yaml")
        print(config.model.hidden_dim)  # 256
        print(config.training.learning_rate)  # 0.001
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self._config = load_config(config_path)
        
        # Convert nested dicts to Config objects for dot notation access
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> dict[str, Any]:
        """Return config as dictionary."""
        return self._config


class ConfigSection:
    """Helper class for nested config sections."""
    
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)


if __name__ == "__main__":
    # Test config loading
    config = Config()
    
    print("=== Config Test ===")
    print(f"Model hidden_dim: {config.model.hidden_dim}")
    print(f"Training LR: {config.training.learning_rate}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Num classes: {config.model.num_classes}")
    print("Config loaded successfully!")