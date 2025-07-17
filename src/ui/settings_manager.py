import json
import os
from typing import Dict, Any
import logging

class SettingsManager:
    def __init__(self):
        self.settings = {
            'output_directory': os.path.join(os.path.expanduser("~"), "Raw Detections"),
            'confidence_threshold': 0.30,
            'confidence_threshold_enabled': False,  # Whether custom confidence threshold is enabled
            'recording_output_directory': os.path.join(os.path.expanduser("~"), "Recordings"),
            'record_mode': False,  # Default value, but won't be persisted
            'delete_raw_failed_detections': False  # Default value for auto delete raw failed detections
        }
        self.load_settings()
        # Directory creation is now handled in load_settings()

    def load_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Don't load record_mode from file
                    if 'record_mode' in loaded_settings:
                        del loaded_settings['record_mode']
                    # Update settings with loaded values (including confidence_threshold if enabled)
                    self.settings.update(loaded_settings)
                    
                    # Ensure directories exist after loading
                    if 'output_directory' in loaded_settings:
                        os.makedirs(loaded_settings['output_directory'], exist_ok=True)
                    if 'recording_output_directory' in loaded_settings:
                        os.makedirs(loaded_settings['recording_output_directory'], exist_ok=True)
                        
                    logging.info(f"Loaded settings: {self.settings}")
            else:
                # If no settings file exists, create directories with default values
                os.makedirs(self.settings['output_directory'], exist_ok=True)
                os.makedirs(self.settings['recording_output_directory'], exist_ok=True)
                logging.info("No settings file found, using defaults")
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            # Fallback to defaults if loading fails
            os.makedirs(self.settings['output_directory'], exist_ok=True)
            os.makedirs(self.settings['recording_output_directory'], exist_ok=True)

    def save_settings(self):
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".road_defect_settings.json")
            # Create a copy of settings without record_mode
            settings_to_save = self.settings.copy()
            if 'record_mode' in settings_to_save:
                del settings_to_save['record_mode']
            # Save confidence_threshold and its enabled state
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
        
        # Validate directory paths
        if key in ['output_directory', 'recording_output_directory']:
            if value and not os.path.isabs(value):
                # Convert relative paths to absolute paths
                value = os.path.abspath(value)
            if value:
                try:
                    os.makedirs(value, exist_ok=True)
                    logging.info(f"Created/verified directory: {value}")
                except Exception as e:
                    logging.error(f"Failed to create directory {value}: {e}")
                    # Don't update the setting if directory creation fails
                    return False
        
        self.settings[key] = value
        
        # Only save if it's not the record_mode setting
        if key != 'record_mode':
            self.save_settings()
        
        return True

    def get_confidence_threshold(self):
        # If confidence threshold is disabled, always return 0.30
        if not self.settings.get('confidence_threshold_enabled', False):
            return 0.30
        # If enabled, return the saved value or default to 0.30
        return self.settings.get('confidence_threshold', 0.30)
    
    def set_confidence_threshold_enabled(self, enabled: bool):
        """Enable or disable custom confidence threshold"""
        self.settings['confidence_threshold_enabled'] = enabled
        if not enabled:
            # When disabled, reset to default but don't change the saved custom value
            logging.info("Confidence threshold disabled, using default 0.30")
        else:
            logging.info(f"Confidence threshold enabled, using custom value: {self.settings.get('confidence_threshold', 0.30)}")
        self.save_settings()
    
    def is_confidence_threshold_enabled(self):
        """Check if custom confidence threshold is enabled"""
        return self.settings.get('confidence_threshold_enabled', False)
    
    def get_output_directory(self):
        """Get the output directory, ensuring it exists"""
        output_dir = self.settings.get('output_directory', os.path.join(os.path.expanduser("~"), "Raw Detections"))
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create output directory {output_dir}: {e}")
            # Fallback to default
            output_dir = os.path.join(os.path.expanduser("~"), "Raw Detections")
            os.makedirs(output_dir, exist_ok=True)
            self.settings['output_directory'] = output_dir
        return output_dir
    
    def get_recording_directory(self):
        """Get the recording directory, ensuring it exists"""
        rec_dir = self.settings.get('recording_output_directory', os.path.join(os.path.expanduser("~"), "Recordings"))
        try:
            os.makedirs(rec_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create recording directory {rec_dir}: {e}")
            # Fallback to default
            rec_dir = os.path.join(os.path.expanduser("~"), "Recordings")
            os.makedirs(rec_dir, exist_ok=True)
            self.settings['recording_output_directory'] = rec_dir
        return rec_dir