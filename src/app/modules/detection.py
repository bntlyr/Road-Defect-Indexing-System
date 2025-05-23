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
from piexif.helper import UserComment
from src.app.modules.gps_reader import GPSReader
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional, Tuple

# Set up logging for error reporting and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectDetector:
    def __init__(self, model_path, save_dir=None, gps_reader=None):
        # Force CPU usage
        self.device = 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with CPU optimizations
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Set model parameters for CPU optimization
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.max_det = 50 # Maximum detections per image
        self.model.agnostic = True  # Class-agnostic NMS
        self.model.multi_label = False  # Single-label per box
        self.model.verbose = False  # Disable verbose output
        
        # Enable CPU optimizations
        torch.set_num_threads(os.cpu_count())  # Use all CPU cores
        torch.set_num_interop_threads(os.cpu_count())  # Use all CPU cores for inter-op
        
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole']
        
        # Set default save directory
        self.save_dir = save_dir or os.path.join(os.path.expanduser("~"), "RDI-Detections")
        os.makedirs(self.save_dir, exist_ok=True)
        self.confidence_threshold = 0.25
        
        # Use provided GPS reader or create a new one
        self.gps_reader = gps_reader
        self.last_gps_update = 0
        self.gps_update_interval = 1.0  # Update GPS data every second
        
        # Color mapping for different defect types (BGR format)
        self.defect_colors = {
            'Linear-Crack': (255, 105, 180),  # Pink
            'Alligator-Crack': (128, 0, 128),  # Purple
            'Pothole': (0, 0, 255),  # Blue
        }
        
        # Initialize counts and queues
        self.frame_counts = {name: 0 for name in self.class_names}
        self.defect_counts = {}
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Pre-allocate buffers for better performance
        self.input_size = (640, 640)
        self.input_buffer = np.zeros((*self.input_size, 3), dtype=np.uint8)
        
        # Cache for processed frames
        self.frame_cache = {}
        self.cache_size = 10  # Keep last 10 frames

    def _save_worker(self):
        """Background worker for saving detections"""
        while True:
            try:
                frame, metadata = self.save_queue.get()
                if frame is None:  # Poison pill
                    break
                self._save_detection(frame, metadata)
                self.save_queue.task_done()
            except Exception as e:
                logging.error(f"Error in save worker: {e}")

    @staticmethod
    def _convert_to_exif_gps(coord):
        """Convert decimal GPS coordinate into EXIF (deg, min, sec) rational format."""
        try:
            degrees = int(coord)
            minutes_float = abs((coord - degrees) * 60)
            minutes = int(minutes_float)
            seconds = int((minutes_float - minutes) * 60 * 10000)
            
            # Validate the conversion
            if not (0 <= minutes < 60 and 0 <= seconds < 600000):
                raise ValueError(f"Invalid minutes/seconds: {minutes}, {seconds}")
                
            return ((abs(degrees), 1), (minutes, 1), (seconds, 10000))
        except Exception as e:
            logging.error(f"Error converting GPS coordinate {coord}: {e}")
            raise

    def _verify_exif_gps(self, exif_dict):
        """Verify that GPS data was properly added to EXIF dictionary."""
        try:
            if 'GPS' not in exif_dict:
                return False
                
            required_tags = [
                piexif.GPSIFD.GPSLatitudeRef,
                piexif.GPSIFD.GPSLatitude,
                piexif.GPSIFD.GPSLongitudeRef,
                piexif.GPSIFD.GPSLongitude
            ]
            
            for tag in required_tags:
                if tag not in exif_dict['GPS']:
                    logging.error(f"Missing required GPS tag: {tag}")
                    return False
                    
            # Verify latitude/longitude values are in correct format
            lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
            lon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
            
            if not (isinstance(lat, tuple) and len(lat) == 3 and
                    isinstance(lon, tuple) and len(lon) == 3):
                logging.error("Invalid GPS coordinate format in EXIF")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error verifying EXIF GPS data: {e}")
            return False

    def _get_valid_gps_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get valid GPS data with validation"""
        if not self.gps_reader:
            logging.warning("No GPS reader available")
            return None, None
            
        current_time = time.time()
        
        # Only try to get new data if enough time has passed
        if current_time - self.last_gps_update >= self.gps_update_interval:
            try:
                lat, lon = self.gps_reader.get_gps_data()
                logging.info(f"GPS Data retrieved - Lat: {lat}, Lon: {lon}")
                if lat is not None and lon is not None:
                    self.last_gps_update = current_time
                    return lat, lon
                else:
                    logging.warning("GPS data returned None values")
            except Exception as e:
                logging.error(f"Error getting GPS data: {e}")
        else:
            logging.debug(f"Using cached GPS data, last update: {current_time - self.last_gps_update:.2f}s ago")
        
        # Return None if no valid data
        return None, None

    def _save_detection(self, frame, metadata):
        """Internal method to save detection with validated GPS data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)

            # Convert frame to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Get GPS data from metadata
            location = metadata.get("Location", {})
            lat = location.get("latitude")
            lon = location.get("longitude")

            # Prepare EXIF dictionary
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # Add GPS data if available and valid
            if lat is not None and lon is not None:
                try:
                    # Validate coordinates
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        # Convert coordinates to EXIF format
                        lat_exif = self._convert_to_exif_gps(abs(lat))
                        lon_exif = self._convert_to_exif_gps(abs(lon))
                        
                        # Add GPS data to EXIF
                        exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
                        exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = lat_exif
                        exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
                        exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = lon_exif
                        exif_dict['GPS'][piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
                        exif_dict['GPS'][piexif.GPSIFD.GPSTimeStamp] = (
                            (int(timestamp[8:10]), 1),  # Hour
                            (int(timestamp[10:12]), 1), # Minute
                            (int(timestamp[12:14]), 1)  # Second
                        )
                        exif_dict['GPS'][piexif.GPSIFD.GPSDateStamp] = f"{timestamp[0:4]}:{timestamp[4:6]}:{timestamp[6:8]}"
                        
                        logging.info(f"Added GPS data to EXIF: {lat}, {lon}")
                    else:
                        logging.warning(f"Invalid GPS coordinates: {lat}, {lon}")
                except Exception as e:
                    logging.error(f"Error adding GPS data to EXIF: {e}")

            # Add defect data as JSON UserComment
            try:
                # Ensure metadata is JSON serializable
                metadata_copy = metadata.copy()
                metadata_copy["Location"] = {
                    "latitude": float(lat) if lat is not None else None,
                    "longitude": float(lon) if lon is not None else None,
                    "timestamp": metadata["Location"]["timestamp"] if metadata["Location"] else None
                }
                
                comment_str = json.dumps(metadata_copy)
                encoded_comment = b"ASCII\0\0\0" + comment_str.encode('utf-8')
                exif_dict['Exif'][piexif.ExifIFD.UserComment] = encoded_comment
                logging.info("Added defect metadata to EXIF")
            except Exception as e:
                logging.error(f"Error encoding defect report: {e}")

            try:
                # Insert EXIF and save
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(image_path, "jpeg", exif=exif_bytes)
                logging.info(f"Image saved successfully with EXIF data: {image_path}")
                
                # Verify EXIF data was saved correctly
                saved_exif = piexif.load(image_path)
                if 'GPS' in saved_exif and piexif.GPSIFD.GPSLatitude in saved_exif['GPS']:
                    logging.info("Verified GPS data in saved image")
                else:
                    logging.warning("GPS data not found in saved image")
                    
            except Exception as e:
                logging.error(f"Error saving image with EXIF: {e}")
                # Fallback to saving without EXIF
                pil_image.save(image_path, "jpeg")
                logging.info(f"Image saved without EXIF data: {image_path}")

        except Exception as e:
            logging.error(f"Error in save_detection: {e}")

    def detect(self, frame):
        """Perform live inference on a frame with GPS validation"""
        # Create a copy of the frame for drawing boxes
        frame_with_boxes = frame.copy()
        
        # Reset frame-specific counts
        for name in self.class_names:
            self.frame_counts[name] = 0
        self.defect_counts.clear()
        
        try:
            # Get validated GPS data
            lat, lon = self._get_valid_gps_data()
            logging.info(f"Detection GPS data - Lat: {lat}, Lon: {lon}, GPS Reader: {self.gps_reader is not None}")
            
            # Use the full frame for detection, only resize for model input
            h, w = frame.shape[:2]
            input_size = (640, 640)  # YOLO model input size
            
            # Calculate scaling factors
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            # Resize for model input while maintaining aspect ratio
            resized = cv2.resize(frame, input_size)
            
            # Run inference with CPU optimizations
            results = self.model(resized,
                               conf=self.confidence_threshold,
                               verbose=False,
                               device=self.device,
                               agnostic_nms=True,
                               max_det=50)
            
            # Process detections
            if results[0].boxes and len(results[0].boxes) > 0:
                # Convert results to CPU
                boxes = results[0].boxes.cpu()
                logging.info(f"Found {len(boxes)} detections")
                
                # Initialize metadata structure with GPS info
                metadata = {
                    "Total Defect Count": len(boxes),
                    "Average Severity Area": 0,
                    "Defects": [],
                    "Location": {
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": datetime.now().isoformat()
                    } if lat is not None and lon is not None else None
                }
                
                # Process boxes in parallel
                futures = []
                for i, box in enumerate(boxes):
                    # Get box coordinates and scale them
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]
                    
                    logging.info(f"Detection {i}: {class_name} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) with confidence {confidence:.2f}")
                    
                    # Process the box with scaled coordinates (only for display)
                    future = self.thread_pool.submit(
                        self._process_box_with_coords,
                        frame_with_boxes,  # Use frame_with_boxes for drawing
                        class_name,
                        confidence,
                        x1, y1, x2, y2
                    )
                    futures.append((future, class_name, confidence, x1, y1, x2, y2))
                
                # Collect results
                total_severity = 0
                for future, class_name, confidence, x1, y1, x2, y2 in futures:
                    box_data = future.result()
                    if box_data:
                        self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                        self.frame_counts[class_name] += 1
                        
                        # Calculate severity area (normalized to frame size)
                        area = (x2 - x1) * (y2 - y1) / (w * h)
                        total_severity += area
                        
                        # Add to metadata
                        metadata["Defects"].append({
                            "key": f"{class_name}_{self.defect_counts[class_name]}",
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "severity_area": float(area),
                            "location": metadata["Location"]  # Include GPS data with each defect
                        })
                
                # Calculate average severity
                if len(metadata["Defects"]) > 0:
                    metadata["Average Severity Area"] = total_severity / len(metadata["Defects"])
                
                # Queue the save operation with the original frame (no boxes) and validated metadata
                if metadata["Location"] is not None:  # Only save if we have GPS data
                    self.save_queue.put((frame, metadata))  # Use original frame without boxes
                    logging.info(f"Queued detection with GPS data: {metadata['Location']}")
                else:
                    logging.warning("Skipping save - No valid GPS data available")
            else:
                logging.debug("No detections found in frame")
            
            return frame_with_boxes, self.frame_counts.copy()  # Return frame with boxes for display
            
        except Exception as e:
            logging.error(f"Error in detection: {e}")
            return frame_with_boxes, {name: 0 for name in self.class_names}

    def _process_box_with_coords(self, frame, class_name, confidence, x1, y1, x2, y2):
        """Process a single detection box with pre-calculated coordinates"""
        try:
            # Get color for this defect type
            color = self.defect_colors.get(class_name, (255, 0, 0))
            
            # Convert coordinates to integers for drawing
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background and text
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Calculate label position
            label_x = x1
            label_y = y1 - 5
            label_bg_y1 = y1 - text_height - 10
            label_bg_y2 = y1
            
            # Draw label background
            cv2.rectangle(frame, 
                         (label_x, label_bg_y1),
                         (label_x + text_width, label_bg_y2),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label,
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            logging.debug(f"Drew box for {class_name} at ({x1}, {y1}, {x2}, {y2})")
            return class_name, confidence, x1, y1, x2, y2
        except Exception as e:
            logging.error(f"Error processing box: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        # Signal save worker to stop
        self.save_queue.put((None, None))
        self.save_thread.join()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear caches
        self.frame_cache.clear()
        self.input_buffer = None

if __name__ == "__main__":
    model_path = r"./src/app/models/road_defect.pt"
    detector = DefectDetector(model_path)

    # Use the Camera class to get frames
    camera = Camera()  # Instantiate the Camera class
    print("Using camera index:", camera.camera_index)

    while True:
        ret, frame = camera.capture.read()  # Use the camera's capture method
        if not ret:
            print("Failed to capture frame.")
            break

        frame_copy, counts = detector.detect(frame)  # Use the detect method

        cv2.imshow("Image", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cleanup()  # Clean up resources
    detector.cleanup()
