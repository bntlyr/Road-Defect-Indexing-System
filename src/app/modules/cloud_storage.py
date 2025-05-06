import os
import logging
from google.cloud import storage
from datetime import datetime
import cv2
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any

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
            
    def upload_detection(self, image: Any, defect_counts: Dict[str, int], frame_counts: Dict[str, int]) -> bool:
        """Upload detection results to cloud storage"""
        if not self.is_initialized:
            self.logger.warning("Cloud storage not initialized - skipping upload")
            return False
            
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create blob names with folder path
            folder_path = self.settings['folder_path'].rstrip('/')
            image_blob_name = f"{folder_path}/detection_{timestamp}.jpg"
            metadata_blob_name = f"{folder_path}/detection_{timestamp}.json"
            
            # Upload image
            image_blob = self.bucket.blob(image_blob_name)
            image_blob.upload_from_string(
                image.tobytes(),
                content_type='image/jpeg'
            )
            
            # Create metadata
            metadata = {
                'timestamp': timestamp,
                'defect_counts': defect_counts,
                'frame_counts': frame_counts,
                'project_id': self.settings['project_id'],
                'region': self.settings['region'],
                'bucket': self.settings['bucket_name'],
                'folder_path': folder_path
            }
            
            # Upload metadata
            metadata_blob = self.bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )
            
            self.logger.info(f"Uploaded detection {timestamp} to cloud storage")
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