import cv2
import queue
import time
import signal
import sys
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import atexit  # Import atexit for cleanup
import logging
import os
import threading
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from src.app.modules.settings_manager import SettingsManager

class DetectionThread(QThread):
    detection_ready = pyqtSignal(np.ndarray, dict, int)  # Signal to emit frame, counts, and fps
    frame_ready = pyqtSignal(np.ndarray)  # Define the frame_ready signal

    def __init__(self, detector, frame_queue, camera):
        super(DetectionThread, self).__init__()  # Call the parent constructor
        self.detector = detector
        self.frame_queue = frame_queue
        self.camera = camera  # Store the camera instance
        self.running = True

    def run(self):
        self.running = True
        logging.info("CameraThread started running")
        
        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                # Get frame with detection from camera
                frame, counts = self.camera.get_frame()
                if frame is None:
                    logging.error("Failed to capture frame")
                    self.running = False
                    break

                # Emit frame for display
                self.frame_ready.emit(frame)  # Emit the frame

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Update every second
                    fps = frame_count
                    frame_count = 0
                    start_time = time.time()
                    self.detection_ready.emit(frame, counts, fps)  # Emit FPS along with frame and counts

            except Exception as e:
                logging.error(f"Camera thread error: {str(e)}")
                self.running = False
                break

            time.sleep(1/30)  # Limit to ~30 FPS

    def stop(self):
        self.running = False
        self.join()

class Camera(QObject):
    detection_ready = pyqtSignal(np.ndarray, dict, int)  # Define the detection_ready signal
    settings_changed = pyqtSignal(float, str)  # Signal for settings changes (confidence, save_dir)

    def __init__(self, camera_index=None, model_path=None):
        super(Camera, self).__init__()  # Call the parent constructor
        
        # Initialize settings manager
        self.settings_manager = SettingsManager()
        
        # Load saved settings or use defaults
        settings = self.settings_manager.settings
        
        # Initialize camera
        if camera_index is None:
            available_cameras = self.get_available_cameras()
            if available_cameras:
                self.camera_index = settings.get('camera_index', available_cameras[0])
            else:
                raise ValueError("No cameras available.")
        else:
            self.camera_index = camera_index
        
        self.capture = cv2.VideoCapture(self.camera_index)
        self.resolutions = self.get_available_resolutions()
        self.max_resolution = max(self.resolutions, key=lambda x: x[0] * x[1])
        self.max_fps = self.get_available_fps()

        # Set the camera to the maximum resolution and FPS
        self.set_resolution(*self.max_resolution)

        # Initialize detection if model path is provided
        self.detecting = False
        self.detector = None
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole']
        self.defect_colors = {
            'Linear-Crack': (0, 0, 255),      # Red
            'Alligator-Crack': (0, 255, 0),    # Green
            'Pothole': (255, 0, 0),           # Blue
        }
        
        # Load settings from settings manager
        self.confidence_threshold = settings.get('confidence_threshold', 0.25)
        self.save_dir = settings.get('save_dir', os.path.join(os.path.expanduser("~"), "RDI-Detections"))
        self.brightness = settings.get('brightness', 50)
        self.exposure = settings.get('exposure', 50)
        self.zoom_factor = settings.get('zoom_factor', 1.0)
        
        # Apply camera settings
        self.set_brightness(self.brightness)
        self.set_exposure(self.exposure)
        self.set_zoom(self.zoom_factor)
        
        self._last_detection_time = time.time()
        self._detection_interval = 1/30  # Target 30 FPS for detection

        # Store the model path as an attribute
        self.model_path = model_path

        if model_path:
            self.initialize_detector(model_path)

        # Initialize frame queue for communication between threads
        self.frame_queue = queue.Queue(maxsize=10)

        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        # Register cleanup function
        atexit.register(self.cleanup)

        # Initialize detection thread
        self.detection_thread = DetectionThread(self.detector, self.frame_queue, self)
        self.detection_thread.detection_ready.connect(self.handle_detection_results)
        self.detection_thread.start()

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def signal_handler(self, signum, frame):
        """Handle termination signals."""
        print("Termination signal received. Cleaning up...")
        self.cleanup()  # Clean up resources
        sys.exit(0)  # Exit the program

    def cleanup(self):
        """Cleanup resources on exit."""
        print("Cleaning up resources...")
        
        # Use tqdm to show a progress bar for cleanup
        for _ in tqdm(range(100), desc="Cleaning up", leave=False):
            time.sleep(0.01)  # Simulate cleanup work

        # Release the camera and destroy all OpenCV windows
        self.capture.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. You can now use the terminal.")

    def get_available_cameras(self):
        """Returns a list of available camera indices."""
        available_cameras = []
        for i in range(2):  # Check the first 2 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def get_available_resolutions(self):
        """Returns a list of supported resolutions."""
        common_resolutions = [(1920, 1080), (1280, 720), (640, 480), (320, 240)]
        supported_resolutions = []

        for width, height in common_resolutions:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_width, actual_height) == (width, height):
                supported_resolutions.append((width, height))

        return supported_resolutions

    def get_available_fps(self):
        """Returns the maximum FPS supported by the camera."""
        common_fps = [30, 60, 120]
        supported_fps = []

        for fps in common_fps:
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            if actual_fps == fps:
                supported_fps.append(actual_fps)

        return max(supported_fps) if supported_fps else 30  # Return max FPS or fallback to 30

    def set_resolution(self, width, height):
        """Sets the camera resolution."""
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_zoom(self, zoom_factor):
        """Sets the zoom factor for the camera."""
        self.zoom_factor = zoom_factor
        self.settings_manager.update_setting('zoom_factor', self.zoom_factor)

    def set_brightness(self, brightness):
        """Try to set camera brightness with better range support."""
        # Try 0–1 scale as fallback
        for scale in [1.0, 255.0, 100.0]:
            value = brightness / scale
            if self.capture.set(cv2.CAP_PROP_BRIGHTNESS, value):
                actual = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
                logging.debug(f"Brightness set to {value} (actual: {actual})")
                self.settings_manager.update_setting('brightness', value)
                return
        logging.error("Failed to set camera brightness.")

    def set_exposure(self, exposure):
        """Try to set camera exposure with better range support."""
        for scale in [1.0, 255.0, 100.0]:
            value = exposure / scale
            if self.capture.set(cv2.CAP_PROP_EXPOSURE, value):
                actual = self.capture.get(cv2.CAP_PROP_EXPOSURE)
                logging.debug(f"Exposure set to {value} (actual: {actual})")
                self.settings_manager.update_setting('exposure', value)
                return
        logging.error("Failed to set camera exposure.")


    def start_stream(self):
        """Starts the video stream."""
        print("Starting video stream...")
        while True:
            ret, frame = self.capture.read()
            if ret:
                cv2.imshow('Camera Stream', frame)
            else:
                print("Failed to capture frame.")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()  # Clean up resources

    def initialize_detector(self, model_path):
        """Initialize detector using DefectDetector"""
        try:
            # Import DefectDetector here to avoid circular import at module level
            from src.app.modules.detection import DefectDetector
            self.detector = DefectDetector(model_path)
            logging.info("DefectDetector initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize DefectDetector: {str(e)}")
            raise

    def preprocess_image(self, frame):
        """Preprocess image for detection"""
        input_size = (640, 640)
        resized = cv2.resize(frame, input_size)
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, input_size, swapRB=True, crop=False)
        return blob, frame.shape[:2]

    def process_frame(self, frame):
        """Process a single frame with detection if enabled"""
        if not self.detecting or not self.detector:
            return frame, {name: 0 for name in self.class_names}

        try:
            # Use DefectDetector to detect defects
            frame_with_boxes, counts = self.detector.detect(frame)
            return frame_with_boxes, counts
            
        except Exception as e:
            logging.error(f"Error in frame processing: {e}")
            return frame, {name: 0 for name in self.class_names}

    def toggle_detection(self, enabled):
        """Toggle detection on/off"""
        self.detecting = enabled
        logging.info(f"Detection {'enabled' if enabled else 'disabled'}")

    def get_frame(self):
        """Get a single frame with optional detection"""
        ret, frame = self.capture.read()
        if not ret:
            return None, {name: 0 for name in self.class_names}

        # Apply zoom
        if self.zoom_factor != 1.0:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
            x1 = max(0, center_x - new_w // 2)
            y1 = max(0, center_y - new_h // 2)
            x2 = min(w, center_x + new_w // 2)
            y2 = min(h, center_y + new_h // 2)
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (w, h))

        # Run detection if enabled
        if self.detecting and self.detector:
            try:
                # Process frame with detection
                frame_with_boxes, counts = self.process_frame(frame)
                return frame_with_boxes, counts
            except Exception as e:
                logging.error(f"Error in detection: {e}")
                return frame, {name: 0 for name in self.class_names}

        # Return the frame and empty counts if detection is disabled
        return frame, {name: 0 for name in self.class_names}

    def handle_detection_results(self, frame_with_boxes, counts, fps):
        # Handle the detection results (e.g., update the dashboard)
        # Emit or process the detection results as needed
        # For example, you could use a signal to send results back to the main thread
        self.detection_ready.emit(frame_with_boxes, counts, fps)

    def update_settings(self, confidence_threshold: float, save_dir: str) -> bool:
        """Update camera and detector settings"""
        try:
            # Validate confidence threshold
            if not 0.0 <= confidence_threshold <= 1.0:
                logging.error(f"Invalid confidence threshold: {confidence_threshold}")
                return False
                
            # Validate save directory
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except Exception as e:
                    logging.error(f"Failed to create save directory: {e}")
                    return False
            
            # Update settings
            self.confidence_threshold = confidence_threshold
            self.save_dir = save_dir
            
            # Update detector settings if available
            if self.detector:
                self.detector.confidence_threshold = confidence_threshold
                self.detector.save_dir = save_dir
            
            # Save settings to file
            self.settings_manager.update_settings({
                'confidence_threshold': confidence_threshold,
                'save_dir': save_dir,
                'camera_index': self.camera_index,
                'brightness': self.brightness,
                'exposure': self.exposure,
                'zoom_factor': self.zoom_factor
            })
            
            # Emit settings changed signal
            self.settings_changed.emit(confidence_threshold, save_dir)
            
            logging.info(f"Settings updated - Confidence: {confidence_threshold}, Save Dir: {save_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update settings: {e}")
            return False

    def get_current_settings(self) -> dict:
        """Get current camera and detector settings"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'save_dir': self.save_dir,
            'camera_index': self.camera_index,
            'brightness': self.brightness,
            'exposure': self.exposure,
            'zoom_factor': self.zoom_factor
        }

# Example usage
if __name__ == "__main__":
    available_cameras = Camera().get_available_cameras()
    print("Available Cameras:", available_cameras)
    
    # Select a camera index (for example, the second camera if available)
    if len(available_cameras) > 1:
        camera_index = available_cameras[1]  # Change this index to select a different camera
    else:
        camera_index = available_cameras[0]  # Fallback to the first camera

    camera = Camera(camera_index)
    print("Selected Camera Index:", camera.camera_index)
    print("Max Resolution:", camera.max_resolution)
    print("Max FPS:", camera.max_fps)
    camera.start_stream()
