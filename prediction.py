import cv2
import numpy as np
from ultralytics import YOLOv10 as YOLO
import os
import time
from datetime import datetime
import torch

class DefectDetector:
    def __init__(self, model_path, save_dir, update_callback=None):
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
        
        self.save_dir = save_dir
        self.last_save_time = 0
        self.save_delay = 0 # seconds
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole', 'Patch']
        self.update_callback = update_callback  # Callback to update GUI
        
        # Initialize counters
        self.frame_counter = 0  # For frame skipping
        self.saved_frame_counter = 0  # For saved frame numbering
        
        # Color mapping for different defect types (BGR format)
        self.defect_colors = {
            'Linear-Crack': (0, 255, 0),      # Green
            'Alligator-Crack': (0, 0, 255),   # Blue
            'Pothole': (255, 0, 0),          # Red
            'Patch': (255, 255, 0)           # Cyan
        }
        
        # Initialize defect counts and last detection times
        self.defect_counts = {
            'Linear-Crack': 0,
            'Alligator-Crack': 0,
            'Pothole': 0,
            'Patch': 0
        }
        self.last_detection_times = {
            'Linear-Crack': 0,
            'Alligator-Crack': 0,
            'Pothole': 0,
            'Patch': 0
        }
        
        # Frame skipping for better performance
        self.frame_skip = 2  # Process every 3rd frame
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def process_frame(self, frame, roi_points=None, gps_data=None):
        # Skip frames for better performance
        self.frame_counter += 1
        if self.frame_counter % (self.frame_skip + 1) != 0:
            # Return empty counts for skipped frames
            return frame, {name: 0 for name in self.class_names}
        
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        current_time = time.time()
        
        # Initialize frame-specific counts (reset for each frame)
        frame_counts = {name: 0 for name in self.class_names}
        detections_to_save = []
        
        # Resize frame for faster processing if it's too large
        height, width = frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280/width, 720/height)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frame_copy = cv2.resize(frame_copy, None, fx=scale, fy=scale)
        
        if roi_points is not None and len(roi_points) == 4:
            # Create ROI mask
            mask = np.zeros_like(frame)
            roi_array = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [roi_array], (255, 255, 255))
            
            # Apply mask to frame
            roi_frame = cv2.bitwise_and(frame, mask)
            
            # Run inference on ROI with optimized settings
            results = self.model(roi_frame, 
                               conf=0.1, 
                               verbose=False, 
                               half=True,
                               device=self.device,
                               agnostic_nms=True,  # Faster NMS
                               max_det=100)  # Limit max detections for speed
        else:
            # Run inference on entire frame with optimized settings
            results = self.model(frame, 
                               conf=0.1, 
                               verbose=False, 
                               half=True,
                               device=self.device,
                               agnostic_nms=True,  # Faster NMS
                               max_det=100)  # Limit max detections for speed
        
        # Process detections
        if results[0].boxes and len(results[0].boxes) > 0:
            # Convert results to CPU for faster processing
            boxes = results[0].boxes.cpu()
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                class_name = self.class_names[class_id]
                
                # Update frame-specific counts (current frame only)
                frame_counts[class_name] += 1
                
                # Check if enough time has passed since last detection of this class
                if current_time - self.last_detection_times[class_name] >= self.save_delay:
                    # Update global counts and last detection time
                    self.defect_counts[class_name] += 1
                    self.last_detection_times[class_name] = current_time
                
                # Store detection for saving (store all detections, not just those that meet the time delay)
                detections_to_save.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
                
                # Get color for this defect type
                color = self.defect_colors[class_name]
                
                # Draw bounding box and label on display frame
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label background
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame_copy, 
                            (int(x1), int(y1) - text_height - 10),
                            (int(x1) + text_width, int(y1)),
                            color, -1)
                
                # Draw label text
                cv2.putText(frame_copy, label,
                          (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save all detections in one image if we have any
            if detections_to_save and current_time - self.last_save_time >= self.save_delay:
                self.save_detection(frame, frame_copy, detections_to_save, gps_data)
                self.last_save_time = current_time
            
            # Update GUI with current frame counts
            if self.update_callback:
                self.update_callback(frame_counts, gps_data)
        else:
            # If no detections, update GUI with zero counts
            if self.update_callback:
                self.update_callback(frame_counts, gps_data)
        
        return frame_copy, frame_counts

    def save_detection(self, original_frame, annotated_frame, detections, gps_data=None):
        # Update last save time
        self.last_save_time = time.time()
        
        # Defect type abbreviations
        defect_abbreviations = {
            'Linear-Crack': 'LC',
            'Alligator-Crack': 'AC',
            'Pothole': 'PH',
            'Patch': 'P'
        }
        
        # Count defects by type
        defect_counts = {name: 0 for name in self.class_names}
        for detection in detections:
            defect_counts[detection['class_name']] += 1
        
        # Generate base filename with timestamp and GPS coordinates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if gps_data and gps_data["latitude"] is not None and gps_data["longitude"] is not None:
            gps_str = f"lat{gps_data['latitude']:.6f}_lon{gps_data['longitude']:.6f}"
        else:
            gps_str = "no_gps"
        
        # Create filename with defect counts
        total_defects = sum(defect_counts.values())
        filename_parts = [
            f"img-{self.saved_frame_counter}",
            f"Detected-{total_defects}",
            f"LC-{defect_counts['Linear-Crack']}",
            f"AC-{defect_counts['Alligator-Crack']}",
            f"PH-{defect_counts['Pothole']}",
            f"P-{defect_counts['Patch']}",
            gps_str
        ]
        base_filename = "_".join(filename_parts)
        
        # Save the full frame with all detections
        full_frame_path = os.path.join(self.save_dir, f"{base_filename}.jpg")
        cv2.imwrite(full_frame_path, annotated_frame)
        print(f"Saved full frame: {full_frame_path}")
        
        # Save each cropped detection
        for i, detection in enumerate(detections):
            class_name = detection['class_name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Add some padding around the detection
            padding = 20
            h, w = original_frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            # Crop the detection region
            cropped = original_frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Create filename for this detection
            defect_type = defect_abbreviations[class_name]
            cropped_filename = f"img-{self.saved_frame_counter}_{defect_type}_{i+1}.jpg"
            cropped_path = os.path.join(self.save_dir, cropped_filename)
            
            # Create a copy of the cropped region for saving
            save_frame = cropped.copy()
            
            # Draw bounding box and label (relative to the cropped region)
            cv2.rectangle(save_frame, 
                         (x1 - x1_pad, y1 - y1_pad), 
                         (x2 - x1_pad, y2 - y1_pad), 
                         (0, 255, 0), 2)
            cv2.putText(save_frame, f"{class_name} {confidence:.2f}", 
                       (x1 - x1_pad, y1 - y1_pad - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add GPS coordinates to the image if available
            if gps_data and gps_data["latitude"] is not None and gps_data["longitude"] is not None:
                gps_text = f"GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}"
                cv2.putText(save_frame, gps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the cropped detection
            cv2.imwrite(cropped_path, save_frame)
            print(f"Saved cropped detection: {cropped_path}")
        
        # Increment the saved frame counter
        self.saved_frame_counter += 1
