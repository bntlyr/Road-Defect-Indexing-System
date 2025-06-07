import json
import os
import logging
from typing import Dict, Any, Optional

class SettingsManager:
    """Manages persistent application settings"""
    
    def __init__(self):
        self.settings_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'settings.json')
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Ensure all default settings exist
                    default_settings = self._get_default_settings()
                    for key, value in default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
        return self._get_default_settings()

    def _get_default_settings(self):
        """Get default settings"""
        return {
            'confidence_threshold': 0.5,
            'output_dir': '',
            'recording_output_dir': '',
            'record_mode': False,
            'camera_index': 0,
            'brightness': 50,
            'exposure': 50,
            'zoom_factor': 1.0
        }

    def save_settings(self):
        """Save settings to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            return False

    def get_setting(self, key):
        """Get a setting value with proper type conversion"""
        default_value = self._get_default_settings().get(key)
        value = self.settings.get(key, default_value)
        
        # Ensure proper type conversion
        if isinstance(default_value, bool):
            return bool(value)
        elif isinstance(default_value, int):
            return int(value)
        elif isinstance(default_value, float):
            return float(value)
        return value

    def set_setting(self, key, value):
        """Set a setting value with type validation"""
        default_value = self._get_default_settings().get(key)
        if default_value is not None:
            # Convert value to the same type as default
            if isinstance(default_value, bool):
                value = bool(value)
            elif isinstance(default_value, int):
                value = int(value)
            elif isinstance(default_value, float):
                value = float(value)
        self.settings[key] = value
        self.save_settings()

    def update_setting(self, key, value):
        """Update a single setting"""
        self.set_setting(key, value)
        return self.save_settings()

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update multiple settings at once
        
        Args:
            settings (Dict[str, Any]): Dictionary of settings to update
            
        Returns:
            bool: True if settings were updated successfully
        """
        try:
            self.settings.update(settings)
            return self.save_settings()
        except Exception as e:
            logging.error(f"Error updating settings: {e}")
            return False

    def reset_settings(self) -> bool:
        """Reset all settings to default values
        
        Returns:
            bool: True if settings were reset successfully
        """
        return self.save_settings() 