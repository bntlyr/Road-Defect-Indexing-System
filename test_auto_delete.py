#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.settings_manager import SettingsManager

def test_auto_delete_setting():
    print("Testing auto-delete raw failed detections setting...")
    
    # Create settings manager
    sm = SettingsManager()
    print(f"Initial settings: {sm.settings}")
    
    # Test the auto-delete setting
    auto_delete = sm.get_setting('delete_raw_failed_detections')
    print(f"Auto-delete setting: {auto_delete}")
    
    # Test setting it to True
    sm.set_setting('delete_raw_failed_detections', True)
    auto_delete_after = sm.get_setting('delete_raw_failed_detections')
    print(f"Auto-delete after setting to True: {auto_delete_after}")
    
    # Test setting it to False
    sm.set_setting('delete_raw_failed_detections', False)
    auto_delete_after_false = sm.get_setting('delete_raw_failed_detections')
    print(f"Auto-delete after setting to False: {auto_delete_after_false}")
    
    # Test loading a fresh instance
    sm2 = SettingsManager()
    auto_delete_fresh = sm2.get_setting('delete_raw_failed_detections')
    print(f"Fresh instance auto-delete setting: {auto_delete_fresh}")

if __name__ == "__main__":
    test_auto_delete_setting()
