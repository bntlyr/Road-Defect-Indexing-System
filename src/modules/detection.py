import cv2
import torch
import numpy as np
from ultralytics import YOLOv10 as YOLO
import os
import json
import logging
from datetime import datetime
from PIL import Image
import threading
import queue
import piexif
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectDetector:
    def __init__(self, model_path, gps_reader=None, gps_queue=None, settings_manager=None, dashboard=None):
        """Initialize the defect detector"""
        self.dashboard = dashboard  # Store the dashboard instance
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        
        # Initialize model as None first
        self.model = None
        
        # Initialize other attributes
        self.gps_reader = gps_reader
        self.gps_queue = gps_queue
        self.is_active = False
        self.is_video_mode = False
        self.video_path = None
        self.gps_log_path = None
        self.gps_data = None
        self.video_time = 0
        self.last_gps_data = None
        self.gps_update_interval = 0.1  # Update GPS data every 0.1 seconds
        self.last_gps_update = 0
        
        # Use provided settings manager
        if settings_manager is None:
            raise ValueError("Settings manager is required")
        self.settings_manager = settings_manager
        logging.info(f"Using settings manager with output directory: {self.settings_manager.get_setting('output_directory')}")
        
        # Initialize GPS data
        self._init_gps_data()
        
        # Load the model last, after all other initializations
        if model_path:
            self.load_model(model_path)
            if self.model is None:
                raise RuntimeError(f"Failed to load model from {model_path}")
            
            # Set model parameters after successful load
            self.model.conf = 0.20  # Lower default confidence threshold
            self.model.iou = 0.45
            
            logging.info("DefectDetector initialized with model: %s", model_path)
        else:
            raise ValueError("Model path is required")

        # Define the correct class names
        self.class_names = {
            0: 'Linear-Crack',
            1: 'Alligator-Crack',
            2: 'pothole'
        }
        
        self.confidence_threshold = 0.20  # Lower default confidence threshold
        
        self.defect_colors = {
            'Linear-Crack': (0, 255, 0),     # Green
            'Alligator-Crack': (0, 165, 255), # Orange
            'pothole': (0, 0, 255)           # Red
        }

        # Initialize frame counts for each class
        self.frame_counts = {name: 0 for name in self.class_names.values()}
        self.defect_counts = {}
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        self.frame_count = 0

    def _save_worker(self):
        while True:
            frame, metadata = self.save_queue.get()
            if frame is None:
                break
            self._save_detection(frame, metadata)
            self.save_queue.task_done()

    def _convert_to_exif_gps(self, value):
        """Convert decimal degrees to EXIF GPS format (degrees, minutes, seconds)"""
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = int(((value - degrees) * 60 - minutes) * 60 * 100)
        return ((degrees, 1), (minutes, 1), (seconds, 100))

    def set_video_mode(self, video_path, gps_log_path=None):
        """Set up GPS log handling for video mode"""
        self.is_video_mode = True
        self.video_path = video_path
        self.gps_log_path = gps_log_path
        self.video_time = 0
        self.last_gps_data = None
        self.video_gps_log = None
        self.video_gps_data = {}
        self.current_video_time = 0
        
        if gps_log_path and os.path.exists(gps_log_path):
            logging.info(f"Loading GPS log file: {gps_log_path}")
            self._load_gps_log(gps_log_path)
            if not self.video_gps_data:
                logging.warning("No valid GPS data found in log file")
                self.is_video_mode = False
        else:
            # Try to find GPS log file based on video filename
            video_dir = os.path.dirname(video_path)
            video_filename = os.path.basename(video_path)
            
            if video_filename.startswith('recording_'):
                timestamp = video_filename.split('_')[1] + '_' + video_filename.split('_')[2].split('.')[0]
                gps_log_filename = f"gps_log_{timestamp}.txt"
                gps_log_path = os.path.join(video_dir, gps_log_filename)
                
                if os.path.exists(gps_log_path):
                    logging.info(f"Loading GPS log file: {gps_log_path}")
                    self._load_gps_log(gps_log_path)
                    if not self.video_gps_data:
                        logging.warning("No valid GPS data found in log file")
                        self.is_video_mode = False
                else:
                    logging.warning(f"GPS log file not found: {gps_log_path}")
                    self.is_video_mode = False

    def _load_gps_log(self, log_path):
        """Load GPS data from log file"""
        try:
            with open(log_path, 'r') as f:
                # Skip header line
                next(f)
                for line in f:
                    try:
                        # Parse timestamp and GPS data
                        timestamp_str, lat, lon = line.strip().split(',')
                        # Convert timestamp to float (it's in seconds)
                        timestamp = float(timestamp_str)
                        lat = float(lat)
                        lon = float(lon)
                        
                        # Validate GPS coordinates
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            self.video_gps_data[timestamp] = (lat, lon)
                            logging.debug(f"Loaded GPS data: time={timestamp:.3f}s, lat={lat}, lon={lon}")
                        else:
                            logging.warning(f"Invalid GPS coordinates in log: lat={lat}, lon={lon}")
                    except ValueError as e:
                        logging.error(f"Error parsing GPS log line: {line.strip()}, Error: {e}")
                        continue
            logging.info(f"Loaded {len(self.video_gps_data)} GPS entries from log file")
            if not self.video_gps_data:
                logging.warning("No valid GPS data found in log file")
                self.is_video_mode = False
            else:
                # Log the time range of GPS data
                min_time = min(self.video_gps_data.keys())
                max_time = max(self.video_gps_data.keys())
                logging.info(f"GPS data time range: {min_time:.3f}s to {max_time:.3f}s")
        except Exception as e:
            logging.error(f"Error loading GPS log file: {e}")
            self.is_video_mode = False

    def update_video_time(self, current_time):
        """Update current video time for GPS lookup"""
        self.current_video_time = current_time
        # Get and log GPS data for debugging
        lat, lon = self._get_video_gps_data()
        if lat is not None and lon is not None:
            logging.debug(f"GPS data at {current_time:.3f}s: lat={lat}, lon={lon}")
        else:
            logging.debug(f"No GPS data at {current_time:.3f}s")

    def _get_valid_gps_data(self):
        """Get GPS data based on current mode"""
        if self.is_video_mode:
            return self._get_video_gps_data()
        else:
            return self._get_live_gps_data()

    def _get_video_gps_data(self):
        """Get GPS data from video log file"""
        if not self.video_gps_data:
            logging.warning("No valid GPS data available, returning None for GPS.")
            return None, None  # Return default values when no GPS data is available

        # Find the closest timestamp in the GPS data
        timestamps = list(self.video_gps_data.keys())
        if not timestamps:
            return self.last_gps_data  # Return last known GPS data if available

        # Find the closest timestamp
        closest_time = min(timestamps, key=lambda x: abs(x - self.current_video_time))
        
        # If the closest time is within 0.1 seconds (100ms), use it
        if abs(closest_time - self.current_video_time) <= 0.1:
            lat, lon = self.video_gps_data[closest_time]
            self.last_gps_data = (lat, lon)  # Update last known GPS data
            logging.debug(f"Found GPS data at time {closest_time:.3f}s (video time: {self.current_video_time:.3f}s): lat={lat}, lon={lon}")
            return lat, lon
        
        # If no close match, return last known GPS data
        return self.last_gps_data

    def _get_live_gps_data(self):
        """Get GPS data from live GPS reader"""
        current_time = time.time()
        if current_time - self.last_gps_update >= self.gps_update_interval:
            try:
                if self.gps_queue and not self.gps_queue.empty():
                    # Get the most recent GPS data
                    gps_data = None
                    while not self.gps_queue.empty():
                        gps_data = self.gps_queue.get_nowait()
                    if gps_data and len(gps_data) >= 2:
                        self.last_gps_update = current_time
                        self.last_gps_data = (gps_data[0], gps_data[1])
                        return gps_data[0], gps_data[1]
            except queue.Empty:
                pass
        return self.last_gps_data  # Return last known GPS data if available

    def _save_detection(self, frame, detections, gps_data=None):
        """Save detection results with GPS data and metadata in EXIF"""
        try:
            # Get output directory from settings
            output_dir = self.settings_manager.get_setting('output_directory')
            if not output_dir:
                logging.error("No output directory set in settings")
                return None
                
            # Ensure the directory exists
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Using output directory from settings: {output_dir}")
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare metadata
            metadata = {
                'timestamp': timestamp,
                'detections': detections
            }
            
            # Convert frame to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Create EXIF data
            exif = {
                "0th": {},
                "Exif": {
                    piexif.ExifIFD.DateTimeOriginal: datetime.now().strftime("%Y:%m:%d %H:%M:%S")
                },
                "GPS": {},
                "1st": {},
                "thumbnail": None
            }
            
            # Add UserComment with proper encoding
            metadata_str = json.dumps(metadata, indent=2)
            exif["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\0\0\0" + metadata_str.encode('ascii', 'replace')
            
            # Add GPS data if available
            if gps_data and 'latitude' in gps_data and 'longitude' in gps_data:
                lat = gps_data['latitude']
                lon = gps_data['longitude']
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    # Convert to EXIF GPS format
                    lat_exif = self._convert_to_exif_gps(abs(lat))
                    lon_exif = self._convert_to_exif_gps(abs(lon))
                    
                    # Set GPS data
                    exif['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
                    exif['GPS'][piexif.GPSIFD.GPSLatitude] = lat_exif
                    exif['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
                    exif['GPS'][piexif.GPSIFD.GPSLongitude] = lon_exif
                    exif['GPS'][piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
                    
                    # Add timestamp
                    gps_time = datetime.now()
                    exif['GPS'][piexif.GPSIFD.GPSDateStamp] = gps_time.strftime("%Y:%m:%d")
                    exif['GPS'][piexif.GPSIFD.GPSTimeStamp] = (
                        (gps_time.hour, 1),
                        (gps_time.minute, 1),
                        (gps_time.second, 1)
                    )
                    
                    logging.info(f"Added GPS data to EXIF: lat={lat}, lon={lon}")
            
            # Save image with EXIF data
            image_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            
            # Convert EXIF dict to bytes
            exif_bytes = piexif.dump(exif)
            
            # Save image with EXIF data
            pil_image.save(image_path, exif=exif_bytes, quality=95)
            
            # Verify GPS data was saved
            try:
                with Image.open(image_path) as img:
                    if 'GPSInfo' in img.info:
                        logging.info("Verified GPS data was saved in EXIF")
                    else:
                        logging.warning("GPS data was not saved in EXIF")
            except Exception as e:
                logging.error(f"Error verifying GPS data: {e}")
            
            logging.info(f"Saved detection image with metadata to: {image_path}")
            
            return image_path
            
        except Exception as e:
            logging.error(f"Error saving detection: {e}")
            return None

    def load_model(self, model_path):
        """Load the YOLO model"""
        try:
            self.model = YOLO(model_path).to(self.device)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.model = None
            raise

    def start_detection(self):
        """Start defect detection"""
        self.is_active = True
        logging.info("Defect detection started")
    
    def stop_detection(self):
        """Stop defect detection"""
        self.is_active = False
        logging.info("Defect detection stopped")

    def detect(self, frame):
        """Detect defects in the given frame"""
        if not self.is_active or frame is None:
            logging.debug("Detector not active or frame is None, skipping detection")
            return None
        
        try:
            # Keep a copy of the original frame for saving
            original_frame = frame.copy()
            frame_with_boxes = frame.copy()
            
            # Reset frame counts
            for name in self.class_names.values():
                self.frame_counts[name] = 0
            self.defect_counts.clear()
            
            # Get GPS data
            lat, lon = self._get_valid_gps_data()
            has_valid_gps = lat is not None and lon is not None

            # Ensure frame is in the correct format
            if len(frame.shape) == 2:  # If grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
            
            h, w = frame.shape[:2]
            input_size = (640, 640)
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            # Resize frame while maintaining aspect ratio
            scale = min(input_size[0] / w, input_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            # Create a square canvas of 640x640
            canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            # Place the resized image in the center
            y_offset = (input_size[1] - new_h) // 2
            x_offset = (input_size[0] - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            # Run detection
            results = self.model(canvas)
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale coordinates back to original image size
                        x1 = (x1 - x_offset) / scale
                        y1 = (y1 - y_offset) / scale
                        x2 = (x2 - x_offset) / scale
                        y2 = (y2 - y_offset) / scale
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Only process if confidence is above threshold
                        if conf >= self.confidence_threshold:
                            class_name = self.class_names.get(cls, f"Unknown-{cls}")
                            color = self.defect_colors.get(class_name, (255, 255, 255))
                            
                            # Draw box
                            cv2.rectangle(frame_with_boxes, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        color, 2)
                            
                            # Create label with class and confidence
                            label = f"{class_name}"
                            
                            # Get text size for background rectangle
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            
                            # Draw background rectangle for text
                            cv2.rectangle(frame_with_boxes, 
                                        (int(x1), int(y1 - text_h - 10)),
                                        (int(x1 + text_w), int(y1)),
                                        color,
                                        -1)
                            
                            # Draw text in white
                            cv2.putText(frame_with_boxes,
                                      label,
                                      (int(x1), int(y1 - 5)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (255, 255, 255),
                                      2,
                                      cv2.LINE_AA)

                            self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                            self.frame_counts[class_name] += 1

                    # Save detection if we have GPS data
                    if has_valid_gps:
                        location_metadata = {
                            'Location': f"{lat:.6f}, {lon:.6f}",
                            'Timestamp': datetime.now().isoformat(),
                            'GPS': [lat, lon]
                        }
                        # Save the original frame without boxes
                        self.save_queue.put((original_frame, location_metadata))
                        logging.info(f"Queued original image with GPS metadata: {location_metadata['Location']}")
                    else:
                        logging.warning("Skipping save - No GPS data available")

            return frame_with_boxes, self.frame_counts.copy()
            
        except Exception as e:
            logging.error(f"Error in detection: {str(e)}")
            return None

    def cleanup(self):
        self.save_queue.put((None, None))
        self.save_thread.join()
        self.thread_pool.shutdown(wait=True)

    def _init_gps_data(self):
        """Initialize GPS data"""
        self.video_gps_log = None
        self.video_gps_data = {}
        self.current_video_time = 0
        self.video_time = 0
        self.last_gps_data = None
        self.last_gps_update = 0
        self.gps_update_interval = 0.1  # Update GPS data every 0.1 seconds



if __name__ == "__main__":
    from camera import Camera

    class MockGPSReader:
        def get_gps_data(self):
            # Simulated GPS coordinates (e.g., San Francisco)
            return 37.7749, -122.4194

    # Replace with the actual path to your YOLOv10 model
    model_path = "C:/Users/bentl/Desktop/RoadDefectSystem/src/models/road_defect.pt"

    # Initialize the detector
    detector = DefectDetector(model_path=model_path, gps_reader=MockGPSReader())

    try:
        # Initialize your camera
        camera = Camera()
        print(f"Camera initialized with resolution: {camera.max_resolution} at {camera.max_fps} FPS")
        camera.set_zoom(1.0)
        camera.set_flipped(vertical=False, horizontal=False)

        print("Press 'q' to exit the test loop.")

        while True:
            ret, frame = camera.capture.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Apply digital zoom and flipping if set
            frame = camera.digital_zoom(frame, camera.zoom_factor)
            frame = camera.flip_frame(frame, camera.flip_vertical, camera.flip_horizontal)

            # Run detection
            processed_frame, counts = detector.detect(frame)

            # Display the frame
            cv2.imshow("Detections", processed_frame)
            print("Defect counts:", counts)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Exception occurred: {e}")

    finally:
        detector.cleanup()
        camera.cleanup()
