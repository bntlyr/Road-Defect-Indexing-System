import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the ConfigManager with a configuration file.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = self._load_default_config()
        self.load_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration settings.
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            "camera": {
                "width": 1280,
                "height": 720,
                "fps": 30,
                "device_id": 0
            },
            "detection": {
                "model_path": "models/best.pt",
                "confidence_threshold": 0.5,
                "save_dir": "detections",
                "skip_frames": 2
            },
            "gps": {
                "baudrate": 9600,
                "timeout": 1
            },
            "map": {
                "save_dir": "maps",
                "zoom_level": 15
            },
            "ui": {
                "theme": "dark",
                "color_theme": "blue",
                "window_size": {
                    "width": 1280,
                    "height": 800
                }
            }
        }

    def load_config(self) -> None:
        """
        Load configuration from file, falling back to defaults if file doesn't exist.
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self._update_config(loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")

    def save_config(self) -> None:
        """
        Save current configuration to file.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def set_value(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values while preserving defaults.
        
        Args:
            new_config: New configuration values
        """
        def update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v

        update_dict(self.config, new_config)

    def get_camera_config(self) -> Dict[str, Any]:
        """
        Get camera configuration.
        
        Returns:
            Dictionary containing camera settings
        """
        return self.config["camera"]

    def get_detection_config(self) -> Dict[str, Any]:
        """
        Get detection configuration.
        
        Returns:
            Dictionary containing detection settings
        """
        return self.config["detection"]

    def get_gps_config(self) -> Dict[str, Any]:
        """
        Get GPS configuration.
        
        Returns:
            Dictionary containing GPS settings
        """
        return self.config["gps"]

    def get_map_config(self) -> Dict[str, Any]:
        """
        Get map configuration.
        
        Returns:
            Dictionary containing map settings
        """
        return self.config["map"]

    def get_ui_config(self) -> Dict[str, Any]:
        """
        Get UI configuration.
        
        Returns:
            Dictionary containing UI settings
        """
        return self.config["ui"] 