import cv2
import numpy as np
from ultralytics import YOLOv10 as YOLO
import os
import time
from datetime import datetime

class DefectDetector:
    def __init__(self, model_path, save_dir):
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        self.last_save_time = 0
        self.save_delay = 1  # seconds
        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole', 'Patch']
        
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
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def process_frame(self, frame, roi_points=None):
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        current_time = time.time()
        
        if roi_points is not None and len(roi_points) == 4:
            # Create ROI mask
            mask = np.zeros_like(frame)
            roi_array = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [roi_array], (255, 255, 255))
            
            # Apply mask to frame
            roi_frame = cv2.bitwise_and(frame, mask)
            
            # Run inference on ROI
            results = self.model(roi_frame, conf=0.1)
            
            # Draw ROI polygon
            cv2.polylines(frame_copy, [roi_array], True, (0, 255, 0), 2)
        else:
            # Run inference on entire frame
            results = self.model(frame, conf=0.1)
        
        # Process detections
        if results[0].boxes and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                class_name = self.class_names[class_id]
                
                # Check if enough time has passed since last detection of this class
                if current_time - self.last_detection_times[class_name] >= self.save_delay:
                    # Update defect counts and last detection time
                    self.defect_counts[class_name] += 1
                    self.last_detection_times[class_name] = current_time
                    
                    # Save the detection
                    self.save_detection(frame, class_name, confidence, (int(x1), int(y1), int(x2), int(y2)))
                
                # Draw bounding box and label on display frame
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{class_name} {confidence:.2f}", 
                          (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy, self.defect_counts

    def save_detection(self, frame, class_name, confidence, bbox):
        # Generate filename with timestamp and GPS placeholder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{confidence:.2f}_{timestamp}_GPS_placeholder.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Create a copy of the frame for saving
        save_frame = frame.copy()
        
        # Draw bounding box and label on the copy
        x1, y1, x2, y2 = bbox
        cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(save_frame, f"{class_name} {confidence:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the frame with bounding boxes but original colors
        cv2.imwrite(filepath, save_frame)
        print(f"Saved detection: {filepath}")
