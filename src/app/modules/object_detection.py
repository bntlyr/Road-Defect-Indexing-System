import cv2
import numpy as np
from ultralytics import YOLOv10 as YOLO
import torch
import os
from datetime import datetime
import json
from PIL import Image, PngImagePlugin
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import piexif
from piexif.helper import UserComment
from openvino.runtime import Core
import time
from threading import Lock

class DefectDetector:
    def __init__(self, model_path, save_dir=None):
        # Initialize OpenVINO runtime
        self.ie = Core()
        
        # Get the OpenVINO model path
        model_dir = os.path.dirname(model_path)
        ov_model_path = os.path.join(model_dir, "last_openvino_model")
        model_xml = os.path.join(ov_model_path, "last.xml")
        
        print(f"Model path: {model_path}")
        print(f"Model directory: {model_dir}")
        print(f"OpenVINO model path: {ov_model_path}")
        print(f"Model XML path: {model_xml}")
        
        # Ensure the model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Export to OpenVINO format if not already exported
        if not os.path.exists(model_xml):
            print("Exporting model to OpenVINO format...")
            try:
                # Ensure the export directory exists
                os.makedirs(ov_model_path, exist_ok=True)
                
                # Export with explicit path and optimizations
                export_path = self.model.export(format="openvino", imgsz=640, half=True)
                print(f"Model exported to: {export_path}")
                
                # Wait a moment for file system to catch up
                time.sleep(1)
                
                # Verify the export
                if not os.path.exists(model_xml):
                    # Try to find the model in the export path
                    export_dir = os.path.dirname(export_path)
                    if os.path.exists(os.path.join(export_dir, "last.xml")):
                        model_xml = os.path.join(export_dir, "last.xml")
                        print(f"Found model at: {model_xml}")
                    else:
                        # List contents of export directory to debug
                        print(f"Contents of {export_dir}:")
                        for file in os.listdir(export_dir):
                            print(f"  - {file}")
                        # List contents of parent directory
                        print(f"Contents of {os.path.dirname(export_dir)}:")
                        for file in os.listdir(os.path.dirname(export_dir)):
                            print(f"  - {file}")
                        raise RuntimeError(f"Model export completed but file not found at {model_xml}")
                
            except Exception as e:
                print(f"Error exporting model: {e}")
                raise RuntimeError(f"Failed to export model to OpenVINO format: {e}")
        
        # Verify the model exists
        if not os.path.exists(model_xml):
            print(f"Searching for model in parent directories...")
            # Try to find the model in parent directories
            current_dir = os.path.dirname(model_path)
            while current_dir and os.path.exists(current_dir):
                potential_path = os.path.join(current_dir, "last_openvino_model", "last.xml")
                if os.path.exists(potential_path):
                    model_xml = potential_path
                    print(f"Found model at: {model_xml}")
                    break
                current_dir = os.path.dirname(current_dir)
            
            if not os.path.exists(model_xml):
                raise RuntimeError(f"OpenVINO model not found at {model_xml}")
        
        print(f"Loading OpenVINO model from {model_xml}")
        # Load the OpenVINO model
        self.ov_model = self.ie.read_model(model_xml)
        
        # Initialize both CPU and GPU models
        self.cpu_model = None
        self.gpu_model = None
        self.current_model = None
        
        # Try to use GPU, fall back to CPU if not available
        try:
            # Check available devices
            available_devices = self.ie.available_devices
            print(f"Available devices: {available_devices}")
            
            # Initialize CPU model
            print("Initializing CPU model...")
            self.cpu_model = self.ie.compile_model(
                model=self.ov_model,
                device_name="CPU",
                config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "NUM_STREAMS": "4"
                }
            )
            
            # Initialize GPU model if available
            gpu_device = next((device for device in available_devices if device.startswith('GPU')), None)
            if gpu_device:
                print(f"Initializing {gpu_device} model...")
                self.gpu_model = self.ie.compile_model(
                    model=self.ov_model,
                    device_name=gpu_device,
                    config={
                        "PERFORMANCE_HINT": "LATENCY",
                        "NUM_STREAMS": "4",
                        "INFERENCE_PRECISION_HINT": "FP16"
                    }
                )
                self.current_model = self.gpu_model
            else:
                self.current_model = self.cpu_model
                
        except Exception as e:
            print(f"Error setting up models, using CPU only: {e}")
            self.current_model = self.cpu_model
        
        # Get input and output layers
        self.input_layer = self.current_model.input(0)
        self.output_layer = self.current_model.output(0)
        
        # Pre-allocate input tensor for better performance
        self.input_tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'pothole']
        
        # Load settings to get save directory and confidence threshold
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
        self.confidence_threshold = 0.25
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    self.save_dir = settings.get('output_path', save_dir or os.path.join(os.path.expanduser("~"), "RoadDefectDetections"))
                    self.confidence_threshold = settings.get('confidence_threshold', 0.25)
            else:
                self.save_dir = save_dir or os.path.join(os.path.expanduser("~"), "RoadDefectDetections")
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.save_dir = save_dir or os.path.join(os.path.expanduser("~"), "RoadDefectDetections")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Color mapping for different defect types (BGR format)
        self.defect_colors = {
            'Linear-Crack': (0, 255, 0),      # Green
            'Alligator-Crack': (255, 0, 0),   # Blue
            'pothole': (0, 0, 255)           # Red
        }
        
        # Pre-allocate tensors for faster processing
        self.frame_counts = {name: 0 for name in self.class_names}
        self.defect_counts = {}
        
        # Initialize save worker
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # Performance monitoring
        self.inference_times = []
        self.processing_times = []
        
        # Locks for thread-safe operations
        self.lock = Lock()
        self.model_lock = Lock()

    def preprocess_image(self, image):
        """Preprocess image for OpenVINO inference with optimizations"""
        # Store original image dimensions
        self.original_height, self.original_width = image.shape[:2]
        
        # Calculate scaling factors
        self.scale_x = self.original_width / 640
        self.scale_y = self.original_height / 640
        
        # Resize image to model input size using INTER_LINEAR for better performance
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB if needed
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        elif resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize and transpose in-place
        self.input_tensor[0] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return self.input_tensor

    def detect(self, frame, gps_data=None):
        """Perform live inference on a frame using OpenVINO"""
        start_time = time.time()
        
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        
        # Preprocess image
        input_tensor = self.preprocess_image(frame)
        
        # Run inference
        with self.model_lock:
            results = self.current_model([input_tensor])[self.output_layer]
        
        # Process results
        predictions = results[0]
        valid_predictions = predictions[predictions[:, 4] > self.confidence_threshold]
        
        # Initialize metadata
        metadata = {
            "Total Defect Count": len(valid_predictions),
            "Average Severity Area": 0,
            "Defects": {},
            "Location": (0.0, 0.0)
        }
        
        if gps_data and gps_data.get("latitude") and gps_data.get("longitude"):
            metadata["Location"] = (gps_data["latitude"], gps_data["longitude"])
        
        # Process predictions
        for pred in valid_predictions:
            x1, y1, x2, y2, confidence, class_id = pred
            
            # Scale bounding box coordinates back to original image size
            x1 = int(x1 * self.scale_x)
            y1 = int(y1 * self.scale_y)
            x2 = int(x2 * self.scale_x)
            y2 = int(y2 * self.scale_y)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, self.original_width))
            y1 = max(0, min(y1, self.original_height))
            x2 = max(0, min(x2, self.original_width))
            y2 = max(0, min(y2, self.original_height))
            
            class_id = int(class_id)
            class_name = self.class_names[class_id]
            
            with self.lock:
                self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                self.frame_counts[class_name] += 1
            
            defect_key = f"{class_name}_{self.defect_counts[class_name]}"
            metadata["Defects"][defect_key] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "severity_area": 0
            }
            
            # Draw on the display copy only
            color = self.defect_colors[class_name]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_copy, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        color, -1)
            cv2.putText(frame_copy, label,
                      (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate and display FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        cv2.putText(frame_copy, f"FPS: {fps:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Queue the save operation with original frame (no bounding boxes)
        try:
            self.save_queue.put((frame, metadata), block=False)
        except queue.Full:
            pass
        
        return frame_copy, self.frame_counts.copy()

    def _save_worker(self):
        """Background worker for saving detections"""
        while True:
            try:
                frame, metadata = self.save_queue.get(timeout=0.1)
                if frame is None:  # Poison pill
                    break
                self._save_detection(frame, metadata)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in save worker: {e}")
                continue

    def cleanup(self):
        """Clean up resources"""
        print("Starting cleanup...")
        
        # Signal save worker to stop
        try:
            self.save_queue.put((None, None), block=False)
        except queue.Full:
            pass
        
        # Wait for save thread with timeout
        self.save_thread.join(timeout=1.0)
        
        print("Cleanup completed")

    def _save_detection(self, frame, metadata):
        """Internal method to save detection (called by worker thread)"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)

            # Convert OpenCV BGR image to RGB and then to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Prepare EXIF dictionary
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # Add GPS data if available
            lat, lon = metadata.get("Location", (0.0, 0.0))
            if lat and lon:
                def _to_deg(value, loc):
                    deg = int(value)
                    min_float = (value - deg) * 60
                    min = int(min_float)
                    sec = round((min_float - min) * 60 * 10000)
                    return ((deg, 1), (min, 1), (sec, 10000)), loc

                lat_data, lat_ref = _to_deg(abs(lat), "N" if lat >= 0 else "S")
                lon_data, lon_ref = _to_deg(abs(lon), "E" if lon >= 0 else "W")

                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = lat_ref
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = lat_data
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = lon_ref
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = lon_data

            # Add defect data as a UserComment
            defect_report = {
                "Total Defect Count": metadata["Total Defect Count"],
                "Average Severity Area": metadata["Average Severity Area"],
                "Defects": metadata["Defects"]
            }
            comment_str = json.dumps(defect_report, indent=2)
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = UserComment.dump(comment_str)

            # Insert EXIF and save
            exif_bytes = piexif.dump(exif_dict)
            pil_image.save(image_path, "jpeg", exif=exif_bytes)

        except Exception as e:
            print(f"Error saving detection: {e}")

    def update_save_dir(self, new_save_dir):
        """Update the save directory"""
        if new_save_dir and os.path.exists(new_save_dir):
            self.save_dir = new_save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            return True
        return False

    def update_settings(self):
        """Update settings from file"""
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    self.confidence_threshold = settings.get('confidence_threshold', 0.25)
        except Exception as e:
            print(f"Error updating settings: {e}")
