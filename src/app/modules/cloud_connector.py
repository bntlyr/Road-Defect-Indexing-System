import os
import logging
from google.cloud import storage
from datetime import datetime
import cv2
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class CloudStorage:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.bucket = None
        self.is_initialized = False
        
        # Load environment variables
        load_dotenv()
        
        # Get settings from environment
        self.settings = {
            'project_id': os.getenv('GOOGLE_PROJECT_ID', ''),
            'bucket_name': os.getenv('GOOGLE_CLOUD_BUCKET_NAME', ''),
            'region': os.getenv('GOOGLE_CLOUD_REGION', ''),
            'folder_path': os.getenv('GOOGLE_CLOUD_FOLDER_PATH', ''),
            'auto_upload': os.getenv('AUTO_UPLOAD', 'false').lower() == 'true'
        }
        
        # Try to initialize, but don't raise error if it fails
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize cloud storage client"""
        try:
            # Check if credentials file exists
            credentials_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.json")
            if not os.path.exists(credentials_path):
                self.logger.warning("Cloud storage not initialized - credentials file not found")
                return False
            
            # Check if required settings are present
            if not all([self.settings['project_id'], self.settings['bucket_name']]):
                self.logger.warning("Cloud storage not initialized - missing required settings")
                return False
            
            # Initialize client
            self.client = storage.Client.from_service_account_json(credentials_path)
            
            # Get bucket
            self.bucket = self.client.bucket(self.settings['bucket_name'])
            
            # Verify bucket exists
            if not self.bucket.exists():
                self.logger.warning(f"Cloud storage not initialized - bucket {self.settings['bucket_name']} does not exist")
                return False
            
            self.is_initialized = True
            self.logger.info("Cloud storage initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Cloud storage not initialized: {e}")
            self.is_initialized = False
            return False
            
    def upload_detection(self, image: Any, defect_counts: Dict[str, int], frame_counts: Dict[str, int], bboxes: Optional[list] = None) -> bool:
        """Upload detection image with metadata embedded in EXIF UserComment and bounding boxes drawn in red."""
        if not self.is_initialized:
            self.logger.warning("Cloud storage not initialized - skipping upload")
            return False

        try:
            from PIL import Image as PILImage
            import piexif
            import io

            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create blob names with folder path
            folder_path = self.settings['folder_path'].rstrip('/')
            image_blob_name = f"{folder_path}/detection_{timestamp}.jpg"

            # Prepare metadata (convert numpy types to Python types)
            def to_python_type(val):
                if hasattr(val, "item"):
                    return val.item()
                if isinstance(val, np.ndarray):
                    return val.tolist()
                return val

            defect_counts_py = {k: to_python_type(v) for k, v in defect_counts.items()}
            frame_counts_py = {k: to_python_type(v) for k, v in frame_counts.items()}

            metadata = {
                'timestamp': timestamp,
                'defect_counts': defect_counts_py,
                'frame_counts': frame_counts_py,
                'project_id': self.settings['project_id'],
                'region': self.settings['region'],
                'bucket': self.settings['bucket_name'],
                'folder_path': folder_path
            }

            # Convert OpenCV image (BGR) to PIL Image (RGB) and draw bounding boxes if provided
            if isinstance(image, np.ndarray):
                image_bgr = image.copy()
                # Draw bounding boxes if provided
                if bboxes:
                    for bbox in bboxes:
                        # bbox: [x1, y1, x2, y2] or dict with keys
                        if isinstance(bbox, dict):
                            if 'bbox' in bbox:
                                x1, y1, x2, y2 = bbox['bbox']
                            else:
                                x1, y1, x2, y2 = bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')
                        else:
                            x1, y1, x2, y2 = bbox
                        # Draw red bounding box (BGR: (0,0,255))
                        cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(image_rgb)
            else:
                pil_img = image

            # Prepare EXIF UserComment with metadata
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            user_comment = json.dumps(metadata)
            encoded_comment = b"ASCII\0\0\0" + user_comment.encode('utf-8')
            import piexif.helper
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = encoded_comment

            exif_bytes = piexif.dump(exif_dict)

            # Save image with EXIF metadata to bytes buffer
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="JPEG", exif=exif_bytes)
            img_bytes.seek(0)

            # Upload image with metadata
            image_blob = self.bucket.blob(image_blob_name)
            image_blob.upload_from_file(img_bytes, content_type='image/jpeg')

            self.logger.info(f"Uploaded detection image with metadata {timestamp} to cloud storage")
            return True

        except Exception as e:
            self.logger.error(f"Error uploading detection: {e}")
            return False
            
    def list_detections(self) -> List[str]:
        """List all detections in cloud storage"""
        if not self.is_initialized:
            self.logger.warning("Cloud storage not initialized - cannot list detections")
            return []
            
        try:
            # List blobs in folder
            folder_path = self.settings['folder_path'].rstrip('/')
            blobs = self.bucket.list_blobs(prefix=folder_path)
            
            # Filter for metadata files
            detections = []
            for blob in blobs:
                if blob.name.endswith('.json'):
                    detections.append(blob.name)
                    
            return sorted(detections)
            
        except Exception as e:
            self.logger.error(f"Error listing detections: {e}")
            return []
            
    def download_detection(self, blob_name: str) -> Tuple[Optional[bytes], Optional[Dict]]:
        """Download detection from cloud storage"""
        if not self.is_initialized:
            self.logger.warning("Cloud storage not initialized - cannot download detection")
            return None, None
            
        try:
            # Get blob
            blob = self.bucket.blob(blob_name)
            
            # Download metadata
            metadata = json.loads(blob.download_as_string())
            
            # Get image blob name
            image_blob_name = blob_name.replace('.json', '.jpg')
            image_blob = self.bucket.blob(image_blob_name)
            
            # Download image
            image_data = image_blob.download_as_string()
            
            return image_data, metadata
            
        except Exception as e:
            self.logger.error(f"Error downloading detection: {e}")
            return None, None
            
    def __del__(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")