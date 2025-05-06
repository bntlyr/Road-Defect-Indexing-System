import os
import shutil
import json
from datetime import datetime
import cv2
import numpy as np
import logging

class FileManager:
    def __init__(self, storage_path=None):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path or os.path.join(os.path.expanduser("~"), "road_defects")
        os.makedirs(self.storage_path, exist_ok=True)
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.storage_path,
            os.path.join(self.storage_path, "images"),
            os.path.join(self.storage_path, "logs"),
            os.path.join(self.storage_path, "severity_results"),
            os.path.join(self.storage_path, "config")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def save_image(self, image, filename=None, subfolder="images"):
        """Save an image to the specified subfolder"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            
        save_path = os.path.join(self.storage_path, subfolder, filename)
        cv2.imwrite(save_path, image)
        return save_path
        
    def save_detection(self, image, metadata):
        """Save detection image and metadata"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create image filename
            image_filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(self.storage_path, image_filename)
            
            # Save image
            cv2.imwrite(image_path, image)
            
            # Create metadata filename
            metadata_filename = f"detection_{timestamp}.json"
            metadata_path = os.path.join(self.storage_path, metadata_filename)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Saved detection to {image_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving detection: {e}")
            return False
            
    def load_detection(self, timestamp):
        """Load detection image and metadata"""
        try:
            # Create filenames
            image_filename = f"detection_{timestamp}.jpg"
            metadata_filename = f"detection_{timestamp}.json"
            
            image_path = os.path.join(self.storage_path, image_filename)
            metadata_path = os.path.join(self.storage_path, metadata_filename)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Failed to load image from {image_path}")
                
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            return image, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading detection: {e}")
            return None, None
            
    def list_detections(self):
        """List all detection files"""
        try:
            detections = []
            for filename in os.listdir(self.storage_path):
                if filename.startswith("detection_") and filename.endswith(".jpg"):
                    timestamp = filename.replace("detection_", "").replace(".jpg", "")
                    detections.append(timestamp)
            return sorted(detections, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing detections: {e}")
            return []
            
    def delete_detection(self, timestamp):
        """Delete detection files"""
        try:
            # Create filenames
            image_filename = f"detection_{timestamp}.jpg"
            metadata_filename = f"detection_{timestamp}.json"
            
            image_path = os.path.join(self.storage_path, image_filename)
            metadata_path = os.path.join(self.storage_path, metadata_filename)
            
            # Delete files
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            self.logger.info(f"Deleted detection {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting detection: {e}")
            return False
        
    def save_severity_result(self, image, severity_data, gps_data=None):
        """Save severity calculation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with GPS data if available
        if gps_data and gps_data.get("latitude") and gps_data.get("longitude"):
            gps_str = f"lat{gps_data['latitude']:.6f}_lon{gps_data['longitude']:.6f}"
        else:
            gps_str = "no_gps"
            
        filename = f"severity_{timestamp}_{gps_str}.jpg"
        
        # Save image
        image_path = self.save_image(image, filename, subfolder="severity_results")
        
        # Save severity data
        severity_path = os.path.join(
            self.storage_path, 
            "severity_results", 
            f"severity_{timestamp}_{gps_str}.json"
        )
        
        with open(severity_path, 'w') as f:
            json.dump(severity_data, f, indent=4)
            
        return image_path, severity_path
        
    def load_image(self, filename, subfolder="images"):
        """Load an image from the specified subfolder"""
        image_path = os.path.join(self.storage_path, subfolder, filename)
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        return None
        
    def get_latest_detections(self, limit=10):
        """Get the most recent detection results"""
        detections_dir = os.path.join(self.storage_path, "logs")
        files = [f for f in os.listdir(detections_dir) if f.endswith('.json')]
        files.sort(reverse=True)
        
        results = []
        for file in files[:limit]:
            with open(os.path.join(detections_dir, file), 'r') as f:
                results.append(json.load(f))
                
        return results
        
    def cleanup_old_files(self, days=30):
        """Remove files older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_date:
                    os.remove(file_path) 