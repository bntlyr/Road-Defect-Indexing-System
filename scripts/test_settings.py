import sys
sys.path.append('src')
from ui.settings_manager import SettingsManager

# Test settings manager initialization and directory methods
try:
    sm = SettingsManager()
    print('Settings Manager initialized successfully')

    # Test directory helper methods
    output_dir = sm.get_output_directory()
    recording_dir = sm.get_recording_directory()

    print(f'Output directory: {output_dir}')
    print(f'Recording directory: {recording_dir}')

    # Test confidence threshold
    conf_enabled = sm.is_confidence_threshold_enabled()
    conf_value = sm.get_confidence_threshold()
    print(f'Confidence threshold enabled: {conf_enabled}')
    print(f'Confidence threshold value: {conf_value}')

    print('All directory helper methods working correctly!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
