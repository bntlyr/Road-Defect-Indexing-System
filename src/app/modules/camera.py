import cv2
import platform
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

class Camera:
    def __init__(self, camera_index: int = 0):
        self.logger = logging.getLogger(__name__)
        self.camera_index = camera_index
        self.cap = None
        self.fps = 30  # Default FPS
        self.is_initialized = False
        self.is_streaming = False
        self.cameras = {}
        self.current_camera_index = camera_index
        self.camera_width = 1280
        self.camera_height = 720
        self.available_resolutions = ["640x480", "1280x720", "1920x1080"]
        
        # Try to initialize
        self.initialize()
        
    @staticmethod
    def get_available_cameras() -> List[Dict[str, any]]:
        """Try to find all available cameras on the system"""
        available_cameras = {}
        system = platform.system()
        
        # Try different backends based on the system
        backends = []
        if system == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        else:
            backends = [cv2.CAP_V4L2, 0]  # V4L2 for Linux, default for others
        
        # Try each backend
        for backend in backends:
            index = 0
            while True:
                try:
                    # Try to open the camera with the current backend
                    cap = cv2.VideoCapture(index, backend)
                    if not cap.isOpened():
                        cap.release()
                        break
                    
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.release()
                        break
                    
                    # If we get here, the camera is working
                    name = f"Camera {index}"
                    if system == "Windows":
                        try:
                            output = subprocess.check_output(
                                'powershell "Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -like \'*camera*\' } | Select-Object -ExpandProperty Name"',
                                shell=True
                            )
                            names = output.decode().strip().split('\r\n')
                            if index < len(names):
                                name = names[index]
                        except:
                            pass
                    
                    available_cameras[index] = name
                    cap.release()
                    index += 1
                except Exception as e:
                    print(f"Error checking camera {index}: {str(e)}")
                    break
        
        if not available_cameras:
            print("No cameras detected!")
            return {0: "Default Camera"}  # Return a default camera if none detected
        
        return available_cameras
        
    def initialize(self) -> bool:
        """Initialize camera with default settings"""
        try:
            # Try to open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False

            # Test if we can read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error(f"Failed to read frame from camera {self.camera_index}")
                self.cap.release()
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual resolution that was set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update stored resolution if different
            if actual_width != self.camera_width or actual_height != self.camera_height:
                self.camera_width = actual_width
                self.camera_height = actual_height
                self.logger.info(f"Camera resolution adjusted to: {actual_width}x{actual_height}")
            
            self.is_initialized = True
            self.is_streaming = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            if self.cap:
                self.cap.release()
            return False
            
    def get_frame(self) -> Optional[any]:
        """Get frame from camera"""
        if not self.is_initialized or self.cap is None:
            # Try to reinitialize if not initialized
            if not self.initialize():
                return None
            
        try:
            # Try to read frame
            ret, frame = self.cap.read()
            
            # If frame read failed, try to recover
            if not ret or frame is None:
                self.logger.warning("Failed to read frame, attempting recovery...")
                # Release and reinitialize camera
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
                
                # Try to set camera properties again
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Try reading frame again
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.logger.error("Failed to recover camera")
                        return None
                else:
                    self.logger.error("Failed to reinitialize camera")
                    return None
                
            # Validate frame dimensions and data
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                self.logger.warning("Invalid frame dimensions")
                return None
                
            # Ensure frame is in correct format
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.warning("Invalid frame format")
                return None
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            # Try to reinitialize camera if frame reading fails
            try:
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.logger.error("Failed to reinitialize camera")
                    return None
                # Set camera properties again
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception as reinit_error:
                self.logger.error(f"Error reinitializing camera: {reinit_error}")
            return None

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
            
    def release(self):
        """Release camera resources"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_initialized = False
            self.is_streaming = False
        except Exception as e:
            self.logger.error(f"Error releasing camera: {e}")
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.release()
        
    def switch_camera(self, camera_index: int) -> bool:
        """Switch to a different camera by index"""
        try:
            # Release current camera if it exists
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Update camera index
            self.camera_index = camera_index
            self.current_camera_index = camera_index
            
            # Initialize new camera
            return self.initialize()
            
        except Exception as e:
            self.logger.error(f"Error switching camera: {e}")
            return False
            
    def change_resolution(self, resolution_str: str) -> bool:
        """Change camera resolution"""
        if not self.is_initialized or self.cap is None:
            return False
            
        try:
            width, height = map(int, resolution_str.split('x'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return True
            
        except Exception as e:
            self.logger.error(f"Error changing resolution: {e}")
            return False
            
    def change_fps(self, fps: int) -> bool:
        """Change camera FPS"""
        if not self.is_initialized or self.cap is None:
            return False
            
        try:
            self.cap.set(cv2.CAP_PROP_FPS, int(fps))
            self.fps = fps
            return True
        except Exception as e:
            self.logger.error(f"Error changing FPS: {e}")
            return False 