import os
import json
from typing import Dict, Any

# Default settings
DEFAULT_SETTINGS = {
    "camera": {
        "device_id": 0,
        "resolution": "1280x720",
        "fps": 30,
        "flip_horizontal": False,
        "flip_vertical": False,
        "brightness": 0,
        "exposure": 0
    },
    "gps": {
        "enabled": False,
        "port": None,
        "baudrate": 9600
    },
    "detection": {
        "confidence_threshold": 0.5,
        "model_path": "best.pt",
        "save_detections": True,
        "class_filter": []  # Empty list means detect all classes
    },
    "severity": {
        "enabled": True,
        "save_results": True,
        "thresholds": {
            "low": 0.25,
            "moderate": 0.5,
            "high": 0.75
        }
    },
    "storage": {
        "base_dir": os.path.join(os.path.expanduser("~"), "RoadDefectDetections"),
        "organize_by": "date",  # "date" or "location"
        "cleanup_days": 30,
        "auto_delete_raw": False
    },
    "cloud": {
        "enabled": False,
        "credentials_path": None,
        "bucket_name": None,
        "folder_path": "detections",
        "auto_upload": False
    },
    "ui": {
        "theme": "dark",
        "map_zoom": 15,
        "default_location": {
            "lat": 14.5995,
            "lon": 120.9842
        },
        "layout": "split"  # "split" or "overlay"
    }
}

def get_settings_path() -> str:
    """Get the path to the settings file"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "settings.json")

def load_settings() -> Dict[str, Any]:
    """Load settings from the settings file"""
    settings_path = get_settings_path()
    
    # If settings file doesn't exist, create it with default settings
    if not os.path.exists(settings_path):
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
        # Update with any missing default settings
        updated = False
        for key, value in DEFAULT_SETTINGS.items():
            if key not in settings:
                settings[key] = value
                updated = True
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in settings[key]:
                        settings[key][subkey] = subvalue
                        updated = True
                
        if updated:
            save_settings(settings)
            
        return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS

def save_settings(settings: Dict[str, Any]) -> bool:
    """Save settings to the settings file"""
    try:
        settings_path = get_settings_path()
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def update_setting(key: str, value: Any) -> bool:
    """Update a single setting"""
    settings = load_settings()
    
    # Split the key by dots to handle nested settings
    keys = key.split('.')
    current = settings
    
    # Navigate to the nested setting
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Update the value
    current[keys[-1]] = value
    
    return save_settings(settings)

def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value"""
    settings = load_settings()
    
    # Split the key by dots to handle nested settings
    keys = key.split('.')
    current = settings
    
    # Navigate to the nested setting
    for k in keys:
        if k not in current:
            return default
        current = current[k]
    
    return current 