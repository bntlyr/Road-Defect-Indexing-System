import json
import os
from typing import Dict, Any
import logging

class SettingsManager:
    def __init__(self):
        self.settings = {
            'output_directory': os.path.join(os.path.expanduser("~"), "Raw Detections"),
            'confidence_threshold': 0.25,
            'cloud_directory': os.path.join(os.path.expanduser("~"), "RoadDefectCloud"),
            'record_mode': False  # Default value, but won't be persisted
        }
        self.load_settings()
        # Ensure output directory exists
        os.makedirs(self.settings['output_directory'], exist_ok=True)

    def load_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Don't load record_mode from file
                    if 'record_mode' in loaded_settings:
                        del loaded_settings['record_mode']
                    # Update settings with loaded values
                    self.settings.update(loaded_settings)
                    logging.info(f"Loaded settings: {self.settings}")
        except Exception as e:
            logging.error(f"Error loading settings: {e}")

    def save_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            # Create a copy of settings without record_mode
            settings_to_save = self.settings.copy()
            if 'record_mode' in settings_to_save:
                del settings_to_save['record_mode']
            with open(settings_file, 'w') as f:
                json.dump(settings_to_save, f, indent=2)
            logging.info(f"Saved settings: {settings_to_save}")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def get_setting(self, key):
        value = self.settings.get(key)
        logging.debug(f"Getting setting {key}: {value}")
        return value

    def set_setting(self, key, value):
        logging.info(f"Setting {key} to {value}")
        self.settings[key] = value
        # Only save if it's not the record_mode setting
        if key != 'record_mode':
            self.save_settings()

    def get_confidence_threshold(self):
        return self.settings.get('confidence_threshold', 0.25)

    def set_confidence_threshold(self, defect_type, value):
        self.settings['confidence_thresholds'][defect_type] = max(0, min(1, value))  # Ensure value is between 0 and 1
        self.save_settings() 