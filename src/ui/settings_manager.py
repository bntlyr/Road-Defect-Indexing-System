import json
import os
from typing import Dict, Any

class SettingsManager:
    def __init__(self):
        self.settings = {
            'output_directory': os.path.join(os.path.expanduser("~"), "Raw Detections"),
            'confidence_threshold': 0.25,
            'cloud_directory': os.path.join(os.path.expanduser("~"), "RoadDefectCloud"),
            'record_mode': False  # Default value, but won't be persisted
        }
        self.load_settings()

    def load_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Don't load record_mode from file
                    if 'record_mode' in loaded_settings:
                        del loaded_settings['record_mode']
                    self.settings.update(loaded_settings)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            # Create a copy of settings without record_mode
            settings_to_save = self.settings.copy()
            if 'record_mode' in settings_to_save:
                del settings_to_save['record_mode']
            with open(settings_file, 'w') as f:
                json.dump(settings_to_save, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value
        # Only save if it's not the record_mode setting
        if key != 'record_mode':
            self.save_settings()

    def get_confidence_threshold(self):
        return self.settings.get('confidence_threshold', 0.25)

    def set_confidence_threshold(self, value: int) -> None:
        """Set confidence threshold as a percentage (0-100)"""
        self.settings['confidence_threshold'] = max(0, min(100, value))
        self.save_settings() 