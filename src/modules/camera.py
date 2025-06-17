import cv2
import numpy as np
import time
import atexit
import signal
import sys
import logging
import os

try:
    import comtypes
    import comtypes.client
    DIRECTSHOW_AVAILABLE = True
except ImportError:
    DIRECTSHOW_AVAILABLE = False

# Suppress OpenCV camera error messages
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = None
        self.is_available = False
        self.zoom_factor = 1.0
        self.flip_vertical = False
        self.flip_horizontal = False
        self.max_fps = 30
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # Target 30 FPS
        
        # Try to initialize camera with DirectShow backend
        try:
            # Set backend to DirectShow
            logging.info(f"Initializing camera with index {camera_index}")
            self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if self.capture.isOpened():
                # Set camera properties for optimal performance
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set reasonable resolution
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.capture.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # Use MJPG codec
                
                # Verify settings
                actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
                if actual_fps <= 0:
                    actual_fps = 30
                self.max_fps = actual_fps
                self.frame_interval = 1.0 / self.max_fps
                
                # Log camera properties
                width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logging.info(f"Camera initialized - Resolution: {width}x{height}, FPS: {self.max_fps}")
                
                # Test frame read
                ret, frame = self.capture.read()
                if ret:
                    logging.info(f"Test frame read successful - Shape: {frame.shape}, Type: {frame.dtype}")
                    self.is_available = True
                else:
                    logging.error("Test frame read failed")
                    self.is_available = False
            else:
                logging.error("Failed to open camera with DirectShow backend")
        except Exception as e:
            logging.error(f"Error initializing camera: {str(e)}")
            self.is_available = False

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)

    def read_frame(self):
        """Read a frame with FPS control"""
        if not self.is_available:
            logging.debug("Camera not available for frame read")
            return None
            
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # If not enough time has passed, return None
        if elapsed < self.frame_interval:
            return None
            
        ret, frame = self.capture.read()
        if ret:
            self.last_frame_time = current_time
            logging.debug(f"Frame read successful - Shape: {frame.shape}, Type: {frame.dtype}")
            return frame
        logging.debug("Frame read failed")
        return None

    def cleanup(self):
        """Clean up camera resources"""
        if self.capture is not None:
            self.capture.release()
            self.is_available = False

    def set_zoom(self, factor):
        """Set digital zoom factor"""
        self.zoom_factor = max(1.0, min(factor, 3.0))  # Limit zoom between 1x and 3x

    def set_flipped(self, vertical=False, horizontal=False):
        """Set flip states"""
        self.flip_vertical = vertical
        self.flip_horizontal = horizontal

    def flip_frame(self, frame, vertical=False, horizontal=False):
        """Flip frame vertically and/or horizontally"""
        if vertical and horizontal:
            return cv2.flip(frame, -1)  # Flip both
        elif vertical:
            return cv2.flip(frame, 0)   # Flip vertically
        elif horizontal:
            return cv2.flip(frame, 1)   # Flip horizontally
        return frame

    def digital_zoom(self, frame, factor):
        """Apply digital zoom to frame"""
        if factor <= 1.0:
            return frame

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Calculate new dimensions
        new_width = int(width / factor)
        new_height = int(height / factor)

        # Calculate crop coordinates
        x1 = center_x - new_width // 2
        y1 = center_y - new_height // 2
        x2 = x1 + new_width
        y2 = y1 + new_height

        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def signal_handler(self, signum, frame):
        """Handle cleanup on system signals"""
        self.cleanup()
        exit(0)


if __name__ == "__main__":
    cam = Camera()  # create camera instance
    
    print(f"Camera initialized with resolution: {cam.capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cam.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)} at {cam.max_fps} FPS")
    cam.set_zoom(1.0)
    cam.set_flipped(vertical=False, horizontal=False)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                print("Failed to grab frame")
                break

            frame = cam.digital_zoom(frame, cam.zoom_factor)
            frame = cam.flip_frame(frame, cam.flip_vertical, cam.flip_horizontal)

            cv2.imshow('Live Camera Test', frame)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print(f"FPS: {frame_count / elapsed:.2f}")
                frame_count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cam.cleanup()


