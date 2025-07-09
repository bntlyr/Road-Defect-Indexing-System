#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.settings_manager import SettingsManager

def test_settings():
    print("Testing SettingsManager...")
    
    # Create settings manager
    sm = SettingsManager()
    print(f"Initial settings: {sm.settings}")
    
    # Test setting and getting values
    sm.set_setting('confidence_threshold', 0.75)
    print(f"After setting confidence: {sm.get_setting('confidence_threshold')}")
    
    sm.set_setting('recording_output_directory', 'C:/test/recordings')
    print(f"After setting recording dir: {sm.get_setting('recording_output_directory')}")
    
    # Test loading a fresh instance
    sm2 = SettingsManager()
    print(f"Fresh instance settings: {sm2.settings}")
    print(f"Fresh instance confidence: {sm2.get_setting('confidence_threshold')}")
    print(f"Fresh instance recording dir: {sm2.get_setting('recording_output_directory')}")

if __name__ == "__main__":
    test_settings()
