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
from src.app.modules.camera import Camera

# Set up logging for error reporting and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectDetector:
    def __init__(self, model_path, save_dir=None):
        # Check for CUDA availability and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with GPU optimizations
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole']
        
        # Set default save directory and confidence threshold
        self.save_dir = save_dir or os.path.join(os.path.expanduser("~"), "RoadDefectDetections")
        os.makedirs(self.save_dir, exist_ok=True)
        self.confidence_threshold = 0.25  # Default value
        
        # Initialize GPS reader
        self.gps_reader = GPSReader()  # Initialize GPS reader
        
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
        degrees = int(coord)
        minutes_float = abs((coord - degrees) * 60)
        minutes = int(minutes_float)
        seconds = int((minutes_float - minutes) * 60 * 10000)

        return ((abs(degrees), 1), (minutes, 1), (seconds, 10000))


    def _save_detection(self, frame, metadata):
        """Internal method to save detection (called by worker thread)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)

            original_frame = frame.copy()
            rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Prepare EXIF dictionary
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # Add GPS data if available
            lat, lon = metadata.get("Location", (None, None))
            if lat is not None and lon is not None:
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = DefectDetector._convert_to_exif_gps(lat)
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = DefectDetector._convert_to_exif_gps(lon)
            else:
                logging.info("No valid GPS data found. Saving without GPS data.")

            # Add defect data as JSON UserComment
            defect_report = {
                "Total Defect Count": metadata.get("Total Defect Count", 0),
                "Average Severity Area": metadata.get("Average Severity Area", 0.0),
                "Defects": metadata.get("Defects", []),
                "Location": metadata.get("Location", (0.0, 0.0))  # Ensure this uses the updated Location
            }
            comment_str = json.dumps(defect_report)
            encoded_comment = b"ASCII\0\0\0" + comment_str.encode('utf-8')
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = encoded_comment

            # Insert EXIF and save
            exif_bytes = piexif.dump(exif_dict)
            pil_image.save(image_path, "jpeg", exif=exif_bytes)
            logging.info(f"Image saved successfully: {image_path}")

        except Exception as e:
            logging.error(f"Error saving detection: {e}")



    def detect(self, frame):
        """Perform live inference on a frame"""
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        
        # Reset frame-specific counts
        for name in self.class_names:
            self.frame_counts[name] = 0
        self.defect_counts.clear()
        
        # Run inference with optimized settings
        results = self.model(frame, 
                             conf=self.confidence_threshold, 
                             verbose=False, 
                             half=True,
                             device=self.device,
                             agnostic_nms=True,  # Faster NMS
                             max_det=100)  # Limit max detections for speed
        
        # Process detections
        if results[0].boxes and len(results[0].boxes) > 0:
            # Convert results to CPU for faster processing
            boxes = results[0].boxes.cpu()
            
            # Initialize metadata structure
            metadata = {
                "Total Defect Count": len(boxes),
                "Average Severity Area": 0,
                "Defects": [],
                "Location": (0.0, 0.0)  # Default coordinates as a tuple
            }
            
            # Update GPS coordinates if available
            gps_data = self.gps_reader.read_gps_data()  # Get GPS data
            logging.info(f"GPS data retrieved: {gps_data}")  # Log the GPS data

            if gps_data != (None, None):  # Check if gps_data is not (None, None)
                logging.info(f"GPS DATA: {gps_data}")  # Log types
                
                try:
                    # Convert GPS data to floats
                    lat, lon = map(float, gps_data)
                    metadata["Location"] = (lat, lon)  # Update metadata with float values
                except (ValueError, TypeError) as e:
                    logging.warning(f"Invalid GPS data: {e}. Setting default coordinates.")
                    metadata["Location"] = (0.0, 0.0)  # Default coordinates
            else:
                logging.warning("No valid GPS data found.")
                metadata["Location"] = (0.0, 0.0)  # Default coordinates

            # Log the metadata before processing boxes
            logging.info(f"Initial metadata: {metadata}")

            # Process all boxes at once for better performance
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[class_id]
                
                # Update counts
                self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                self.frame_counts[class_name] += 1
                
                # Create unique key for this defect
                defect_key = f"{class_name}_{self.defect_counts[class_name]}"
                
                # Add defect to metadata
                metadata["Defects"].append({
                    "key": defect_key,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "severity_area": 0
                })
                
                # Get color for this defect type
                color = self.defect_colors.get(class_name, (255, 0, 0))  # Default to red if not found
                
                # Draw bounding box and label on the display frame
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background and text
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame_copy, 
                              (x1, y1 - text_height - 10),
                              (x1 + text_width, y1),
                              color, -1)
                cv2.putText(frame_copy, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Log the final metadata after processing boxes
            logging.info(f"Final metadata: {metadata}")

            # Queue the save operation with the original frame
            self.save_queue.put((frame, metadata))
        
        return frame_copy, self.frame_counts.copy()

    def cleanup(self):
        """Clean up resources"""
        # Signal save worker to stop
        self.save_queue.put((None, None))
        self.save_thread.join()

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
