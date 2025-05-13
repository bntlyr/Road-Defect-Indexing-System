import cv2
import platform
import logging
import subprocess
from typing import List, Dict, Optional, Tuple
import base64
import numpy as np
import time
import traceback
import threading

logger = logging.getLogger(__name__)

class Camera:
    # Define available resolutions as a class attribute
    AVAILABLE_RESOLUTIONS = {
        "640x480": [30, 60],     # VGA
        "800x600": [30, 60],     # SVGA
        "1280x720": [30, 60],    # 720p
        "1920x1080": [30, 60],   # 1080p
        "2560x1440": [30],       # 2K
        "3840x2160": [30]        # 4K
    }

    # Camera type identifiers - Updated to prioritize USB cameras
    CAMERA_TYPES = {
        'USB': ['usb', 'uvc', 'external'],
        'BUILT_IN': ['webcam', 'integrated', 'camera dfu']  # Moved BUILT_IN to second priority
    }

    # Add class-level cache for camera detection
    _camera_cache = {}
    _last_detection_time = 0
    _detection_cache_time = 30  # Cache camera detection for 30 seconds

    def __init__(self, device_index=0):
        self.device_index = device_index
        self._last_device_index = None  # Add this to track last initialized device
        self.cap = None
        self.is_initialized = False
        self.camera_type = None
        self.camera_name = None
        self.device_id = None
        self.last_frame_time = 0
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        # Default settings
        self.settings = {
            'flip_vertical': False,
            'brightness': 50,
            'exposure': 50,
            'fps': 60,
            'zoom': 1.0
        }
        self.initialize()
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()

    @staticmethod
    def _identify_camera_type(name: str) -> str:
        """Identify if a camera is built-in or USB based on its name"""
        name_lower = name.lower()
        for camera_type, keywords in Camera.CAMERA_TYPES.items():
            if any(keyword in name_lower for keyword in keywords):
                return camera_type
        return 'USB'  # Default to USB if not clearly identified

    @staticmethod
    def _sort_cameras(camera_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Sort cameras to prioritize USB cameras"""
        usb = []
        built_in = []
        
        for name, device_id in camera_list:
            camera_type = Camera._identify_camera_type(name)
            if camera_type == 'USB':
                usb.append((name, device_id))
            else:
                built_in.append((name, device_id))
        
        # Sort each list by name for consistency
        usb.sort(key=lambda x: x[0])
        built_in.sort(key=lambda x: x[0])
        
        # Return USB cameras first, then built-in cameras
        return usb + built_in

    @staticmethod
    def get_available_cameras() -> Dict[int, str]:
        """Get a list of available cameras on the system with strict mapping"""
        # Check cache first
        current_time = time.time()
        if (Camera._camera_cache and 
            current_time - Camera._last_detection_time < Camera._detection_cache_time):
            logger.info("Returning cached camera information")
            return Camera._camera_cache

        available_cameras = {}
        try:
            logger.info("Starting camera detection using direct testing...")
            if platform.system() == "Windows":
                # Test camera indices directly without PowerShell
                camera_names = []
                for index in range(10):  # Test first 10 indices
                    try:
                        logger.info(f"Testing camera index {index}")
                        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                # Get camera properties
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = int(cap.get(cv2.CAP_PROP_FPS))
                                
                                # Try to get a more descriptive name
                                try:
                                    device_name = f"Camera {index}"
                                    try:
                                        output = subprocess.check_output(
                                            f'wmic path Win32_PnPEntity where "DeviceID like \'%VID_%PID_%\'" get Name /value',
                                            shell=True
                                        ).decode().strip()
                                        for line in output.split('\n'):
                                            if 'Name=' in line:
                                                device_name = line.split('=')[1].strip()
                                                break
                                    except:
                                        pass
                                    
                                    camera_names.append((device_name, f"index_{index}"))
                                    logger.info(f"Found camera: {device_name} at index {index}")
                                except:
                                    camera_names.append((f"Camera {index}", f"index_{index}"))
                                    logger.info(f"Found camera at index {index}")
                            cap.release()
                    except Exception as e:
                        logger.debug(f"Error testing camera index {index}: {str(e)}")
                        continue

                # Sort cameras to ensure consistent order
                sorted_cameras = Camera._sort_cameras(camera_names)
                logger.info(f"Sorted cameras: {sorted_cameras}")

                # Test each camera index with strict mapping
                for index, (name, device_id) in enumerate(sorted_cameras):
                    logger.info(f"Testing camera index {index} with name {name}")
                    max_retries = 2
                    for retry in range(max_retries):
                        try:
                            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                            if not cap.isOpened():
                                logger.warning(f"Camera {index} ({name}) not opened, skipping")
                                break

                            # Get camera properties
                            start_time = time.time()
                            while time.time() - start_time < 2:
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                                    
                                    # Get supported resolutions and FPS
                                    supported_resolutions = {}
                                    for res, fps_list in Camera.AVAILABLE_RESOLUTIONS.items():
                                        w, h = map(int, res.split('x'))
                                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        if abs(actual_w - w) <= w * 0.05 and abs(actual_h - h) <= h * 0.05:
                                            supported_resolutions[res] = fps_list
                                    
                                    # Determine camera type
                                    camera_type = Camera._identify_camera_type(name)
                                    
                                    camera_info = {
                                        'name': name,
                                        'device_id': device_id,
                                        'camera_type': camera_type,
                                        'resolution': f"{width}x{height}",
                                        'fps': fps,
                                        'brightness': 50,
                                        'exposure': 50,
                                        'supported_resolutions': supported_resolutions
                                    }
                                    available_cameras[index] = camera_info
                                    logger.info(f"Camera {index} ({name}) properties: {camera_info}")
                                    break
                                time.sleep(0.1)
                            
                            cap.release()
                            break
                        except Exception as e:
                            logger.error(f"Error testing camera {index} ({name}) (attempt {retry + 1}): {str(e)}")
                            if retry == max_retries - 1:
                                break
                            time.sleep(0.5)
                            if cap is not None:
                                cap.release()

            logger.info(f"Found {len(available_cameras)} cameras: {available_cameras}")
            
            if not available_cameras:
                logger.warning("No cameras found, adding default camera")
                available_cameras[0] = {
                    'name': "Default Camera",
                    'device_id': None,
                    'camera_type': 'BUILT_IN',
                    'resolution': "1280x720",
                    'fps': 30,
                    'brightness': 50,
                    'exposure': 50,
                    'supported_resolutions': {"1280x720": [30, 60]}
                }

            # Update cache
            Camera._camera_cache = available_cameras
            Camera._last_detection_time = current_time

        except Exception as e:
            logger.error(f"Error getting available cameras: {str(e)}")
            logger.error(traceback.format_exc())
            available_cameras[0] = {
                'name': "Default Camera",
                'device_id': None,
                'camera_type': 'BUILT_IN',
                'resolution': "1280x720",
                'fps': 30,
                'brightness': 50,
                'exposure': 50,
                'supported_resolutions': {"1280x720": [30, 60]}
            }
            # Update cache even in case of error
            Camera._camera_cache = available_cameras
            Camera._last_detection_time = current_time
        
        return available_cameras
        
    def get_supported_resolutions(self) -> List[str]:
        """Get list of supported resolutions for the current camera"""
        try:
            if not self.is_initialized or self.cap is None:
                logger.warning("Camera not initialized when getting resolutions")
                if not self.initialize():
                    logger.error("Failed to initialize camera when getting resolutions")
                    return []

            supported_resolutions = []
            current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Current camera resolution: {current_width}x{current_height}")

            # Test each resolution
            for resolution in self.AVAILABLE_RESOLUTIONS.keys():
                try:
                    width, height = map(int, resolution.split('x'))
                    logger.debug(f"Testing resolution: {resolution}")

                    # Set MJPG codec for better compatibility with high resolutions
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                    # Verify the resolution was set
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.debug(f"Actual resolution after setting {resolution}: {actual_width}x{actual_height}")

                    # More lenient resolution check (within 5% of target)
                    width_diff = abs(actual_width - width) / width
                    height_diff = abs(actual_height - height) / height
                    if width_diff <= 0.05 and height_diff <= 0.05:
                        supported_resolutions.append(resolution)
                        logger.info(f"Resolution {resolution} is supported")
                except Exception as e:
                    logger.error(f"Error testing resolution {resolution}: {str(e)}")
                    continue

            # Restore original resolution
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
                logger.info("Restored original resolution")
            except Exception as e:
                logger.error(f"Error restoring original resolution: {str(e)}")

            if not supported_resolutions:
                logger.warning("No supported resolutions found, returning default resolutions")
                return list(self.AVAILABLE_RESOLUTIONS.keys())

            return supported_resolutions

        except Exception as e:
            logger.error(f"Unexpected error in get_supported_resolutions: {str(e)}")
            return list(self.AVAILABLE_RESOLUTIONS.keys())

    def initialize(self):
        """Initialize the camera with the specified device index"""
        try:
            # If already initialized with the same device index, skip
            if (self.is_initialized and self.cap is not None and 
                self.device_index == self._last_device_index):
                logger.info("Camera already initialized with same device index, skipping")
                return True

            logger.info(f"Initializing camera at index {self.device_index}")
            
            # Get camera info before initialization
            cameras = self.get_available_cameras()
            if self.device_index in cameras:
                camera_info = cameras[self.device_index]
                self.camera_name = camera_info['name']
                self.device_id = camera_info['device_id']
                self.camera_type = camera_info['camera_type']
                logger.info(f"Initializing {self.camera_type} camera: {self.camera_name}")
            
            # Release any existing camera
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                self.is_initialized = False
            
            # Try to open camera with DirectShow backend on Windows
            if platform.system() == "Windows":
                logger.info("Using DirectShow backend on Windows")
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.device_index} ({self.camera_name})")
                return False

            # Set properties with timeout
            start_time = time.time()
            while time.time() - start_time < 5:
                try:
                    # Set MJPG codec first
                    logger.info("Setting MJPG codec")
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Set initial resolution to 1080p
                    logger.info("Setting initial resolution to 1080p")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
                    # Set FPS
                    logger.info("Setting FPS to 60")
                    self.cap.set(cv2.CAP_PROP_FPS, 60)
                    
                    # Set buffer size to minimum for lower latency
                    logger.info("Setting minimum buffer size")
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Set auto exposure to manual mode and initial values
                    logger.info("Setting exposure to manual mode")
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.5)  # Set to 50%
                    
                    # Set initial brightness
                    logger.info("Setting initial brightness")
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Set to 50%
                    
                    # Verify camera properties
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                        actual_brightness = int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS) * 100)
                        actual_exposure = int(self.cap.get(cv2.CAP_PROP_EXPOSURE) * 100)
                        
                        logger.info(f"Camera initialized with:")
                        logger.info(f"Name: {self.camera_name}")
                        logger.info(f"Type: {self.camera_type}")
                        logger.info(f"Resolution: {actual_width}x{actual_height}")
                        logger.info(f"FPS: {actual_fps}")
                        logger.info(f"Brightness: {actual_brightness}")
                        logger.info(f"Exposure: {actual_exposure}")
                        
                        self.is_initialized = True
                        self._last_device_index = self.device_index  # Store last device index
                        return True
                    
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error during camera initialization attempt: {str(e)}")
                    time.sleep(0.1)
            
            logger.error(f"Camera initialization timed out for {self.camera_name}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
            
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            logger.error(traceback.format_exc())
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            self.is_initialized = False
            return False

    def _capture_frames(self):
        """Background thread to continuously capture frames"""
        while self.is_initialized and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.frame_lock:
                        self.frame_buffer = frame
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.01)  # Short sleep to prevent CPU hogging
            except Exception as e:
                logger.error(f"Error in capture thread: {str(e)}")
                time.sleep(0.1)

    def get_frame(self):
        """Get the latest frame from the buffer"""
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return None

        try:
            with self.frame_lock:
                if self.frame_buffer is None:
                    return None
                
                # Check if frame is too old (more than 1 second)
                if time.time() - self.last_frame_time > 1.0:
                    logger.warning("Frame is too old, reinitializing camera")
                    self.reinitialize()
                    return None

                frame = self.frame_buffer.copy()
                
                # Apply transformations
                if self.settings['flip_vertical']:
                    frame = cv2.flip(frame, 0)
                    
                # Apply zoom if needed
                if self.settings['zoom'] != 1.0:
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # Calculate new dimensions
                    new_w = int(w / self.settings['zoom'])
                    new_h = int(h / self.settings['zoom'])
                    
                    # Calculate crop coordinates
                    x1 = max(0, center_x - new_w // 2)
                    y1 = max(0, center_y - new_h // 2)
                    x2 = min(w, center_x + new_w // 2)
                    y2 = min(h, center_y + new_h // 2)
                    
                    # Crop and resize
                    cropped = frame[y1:y2, x1:x2]
                    frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                
                return frame
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def reinitialize(self):
        """Reinitialize the camera"""
        try:
            self.release()
            time.sleep(0.5)  # Give some time for the camera to reset
            return self.initialize()
        except Exception as e:
            logger.error(f"Error reinitializing camera: {str(e)}")
            return False

    def switch_camera(self, device_index):
        """Switch to a different camera device"""
        logger.info(f"Switching to camera at index {device_index}")
        self.release()
        self.device_index = device_index
        return self.initialize()

    def change_resolution(self, resolution):
        """Change the camera resolution"""
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return False
        try:
            width, height = map(int, resolution.split('x'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Resolution changed to {resolution}")
            return True
        except Exception as e:
            logger.error(f"Error changing resolution: {str(e)}")
            return False

    def release(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            logger.info("Camera released")

    def get_camera_name(self, index: int) -> str:
        """Get camera name from index"""
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output(
                    'powershell "Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -like \'*camera*\' } | Select-Object -ExpandProperty Name"',
                    shell=True
                )
                names = output.decode().strip().split('\r\n')
                if index < len(names):
                    return names[index]
            return f"Camera {index}"
        except:
            return f"Camera {index}"
            
    def change_fps(self, fps: int) -> bool:
        """Change camera FPS"""
        if not self.is_initialized or self.cap is None:
            return False
            
        try:
            self.cap.set(cv2.CAP_PROP_FPS, int(fps))
            return True
        except Exception as e:
            logger.error(f"Error changing FPS: {e}")
            return False

    def stream_frames(self):
        """Stream frames from the camera"""
        while self.is_initialized and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read frame from camera")
                break

            # Convert the frame to base64
            _, buffer = cv2.imencode('.jpg', frame)  # Encode the image as JPEG
            image_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string

            yield image_base64  # Yield the base64-encoded frame

    def update_settings(self, settings: dict) -> bool:
        """Update camera settings with immediate effect"""
        try:
            if not self.is_initialized or self.cap is None:
                logger.error("Camera not initialized when updating settings")
                return False

            # Update settings immediately
            if 'flip_vertical' in settings:
                self.settings['flip_vertical'] = bool(settings['flip_vertical'])
                logger.info(f"Updated flip_vertical setting to {self.settings['flip_vertical']}")

            if 'brightness' in settings:
                brightness = int(settings['brightness'])
                if 0 <= brightness <= 100:
                    self.settings['brightness'] = brightness
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness / 100.0)
                    logger.info(f"Updated brightness setting to {brightness}")

            if 'exposure' in settings:
                exposure = int(settings['exposure'])
                if 0 <= exposure <= 100:
                    self.settings['exposure'] = exposure
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure / 100.0)
                    logger.info(f"Updated exposure setting to {exposure}")

            if 'fps' in settings:
                fps = int(settings['fps'])
                if fps in self.AVAILABLE_RESOLUTIONS:
                    self.cap.set(cv2.CAP_PROP_FPS, fps)
                    self.settings['fps'] = fps
                    logger.info(f"Updated FPS setting to {fps}")

            if 'zoom' in settings:
                zoom = float(settings['zoom'])
                if 0.1 <= zoom <= 5.0:
                    self.settings['zoom'] = zoom
                    logger.info(f"Updated zoom setting to {zoom}x")

            if 'device' in settings:
                device_index = int(settings['device'])
                if device_index != self.device_index:
                    self.device_index = device_index
                    return self.reinitialize()

            return True
        except Exception as e:
            logger.error(f"Error updating camera settings: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Simple test function to check camera access
def test_camera(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    ret, frame = cap.read()
    if ret:
        print("Camera is working!")
        cv2.imshow('Test Frame', frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
    else:
        print("Failed to read frame from camera")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()  # Test the camera when the script is run directly 