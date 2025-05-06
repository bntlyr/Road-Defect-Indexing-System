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

class DefectDetector:
    def __init__(self, model_path, save_dir=None):
        # Check for CUDA availability and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with GPU optimizations
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            # Move model to GPU
            self.model.to(self.device)
            # Enable cuDNN benchmarking for faster inference
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'pothole']
        
        # Load settings to get save directory and confidence threshold
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
        self.confidence_threshold = 0.25  # Default value
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
            'Alligator-Crack': (0, 0, 255),   # Blue
            'pothole': (255, 0, 0)           # Red
        }
        
        # Pre-allocate tensors for faster processing
        self.frame_counts = {name: 0 for name in self.class_names}
        self.defect_counts = {}
        
        # Set up model for faster inference
        self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference
        
        # Initialize thread pool for saving detections
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # Initialize thread pool for processing
        self.process_pool = ThreadPoolExecutor(max_workers=2)
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()

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
                print(f"Error in save worker: {e}")

    def _save_detection(self, frame, metadata):
        """Internal method to save detection (called by worker thread)"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            image_filename = f"detection_{timestamp}.png"
            image_path = os.path.join(self.save_dir, image_filename)
            
            # Convert OpenCV BGR image to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Create PNG info object for metadata
            png_info = PngImagePlugin.PngInfo()
            
            # Format metadata as string
            metadata_str = f"Total Defect Count: {metadata['Total Defect Count']}\n"
            metadata_str += f"Average Severity Area: {metadata['Average Severity Area']}\n"
            metadata_str += "Defects:\n"
            metadata_str += json.dumps(metadata['Defects'], indent=2)
            metadata_str += f"\nLocation: {metadata['Location'][0]}, {metadata['Location'][1]}"
            
            # Add metadata to PNG info
            png_info.add_text("Defect_Report", metadata_str)
            
            # Save image with embedded metadata
            pil_image.save(image_path, "PNG", pnginfo=png_info)
            
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

    def detect(self, frame, gps_data=None):
        """Perform live inference on a frame"""
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        
        # Reset frame-specific counts
        with self.lock:
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
                "Defects": {},
                "Location": (0.0, 0.0)  # Default coordinates
            }
            
            # Update GPS coordinates if available
            if gps_data and gps_data.get("latitude") and gps_data.get("longitude"):
                metadata["Location"] = (gps_data["latitude"], gps_data["longitude"])
            
            # Process all boxes at once for better performance
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[class_id]
                
                # Update counts
                with self.lock:
                    self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                    self.frame_counts[class_name] += 1
                
                # Create unique key for this defect
                defect_key = f"{class_name}_{self.defect_counts[class_name]}"
                
                # Add defect to metadata
                metadata["Defects"][defect_key] = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "severity_area": 0
                }
                
                # Get color for this defect type
                color = self.defect_colors[class_name]
                
                # Draw bounding box and label
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
            
            # Queue the save operation
            self.save_queue.put((frame, metadata))
        
        return frame_copy, self.frame_counts.copy()

    def cleanup(self):
        """Clean up resources"""
        # Signal save worker to stop
        self.save_queue.put((None, None))
        self.save_thread.join()
        self.process_pool.shutdown(wait=True)
