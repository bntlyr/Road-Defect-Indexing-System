import json
import os
import logging
from typing import Dict, Any, Optional

class SettingsManager:
    """Manages persistent application settings"""
    
    def __init__(self, settings_file: str = "app_settings.json"):
        """Initialize settings manager
        
        Args:
            settings_file (str): Path to settings file, defaults to app_settings.json in user's home directory
        """
        self.settings_file = os.path.join(os.path.expanduser("~"), settings_file)
        self.default_settings = {
            'confidence_threshold': 0.25,
            'save_dir': os.path.join(os.path.expanduser("~"), "RDI-Detections"),
            'camera_index': 0,
            'brightness': 50,
            'exposure': 50,
            'zoom_factor': 1.0
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create default if file doesn't exist
        
        Returns:
            Dict[str, Any]: Current settings
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Ensure all default settings exist
                    for key, value in self.default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
            else:
                # Create default settings file
                self.save_settings(self.default_settings)
                return self.default_settings
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            return self.default_settings

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file
        
        Args:
            settings (Dict[str, Any]): Settings to save
            
        Returns:
            bool: True if settings were saved successfully
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            self.settings = settings
            logging.info(f"Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            return False

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value
        
        Args:
            key (str): Setting key
            default (Any): Default value if setting doesn't exist
            
        Returns:
            Any: Setting value or default
        """
        return self.settings.get(key, default)

    def update_setting(self, key: str, value: Any) -> bool:
        """Update a single setting
        
        Args:
            key (str): Setting key
            value (Any): New value
            
        Returns:
            bool: True if setting was updated successfully
        """
        try:
            self.settings[key] = value
            return self.save_settings(self.settings)
        except Exception as e:
            logging.error(f"Error updating setting {key}: {e}")
            return False

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update multiple settings at once
        
        Args:
            settings (Dict[str, Any]): Dictionary of settings to update
            
        Returns:
            bool: True if settings were updated successfully
        """
        try:
            self.settings.update(settings)
            return self.save_settings(self.settings)
        except Exception as e:
            logging.error(f"Error updating settings: {e}")
            return False

    def reset_settings(self) -> bool:
        """Reset all settings to default values
        
        Returns:
            bool: True if settings were reset successfully
        """
        return self.save_settings(self.default_settings) 