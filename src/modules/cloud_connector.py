import os
import logging
from google.cloud import storage
from datetime import datetime
import cv2
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import piexif
from PIL import Image

class CloudConnector:
    def __init__(self, cloud_dir):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize attributes
        self.cloud_dir = cloud_dir
        self.ensure_cloud_directory()

    def ensure_cloud_directory(self):
        """Ensure the cloud directory exists"""
        if not os.path.exists(self.cloud_dir):
            os.makedirs(self.cloud_dir)
            self.logger.info(f"Created cloud directory: {self.cloud_dir}")

    def upload_detections(self, source_dir):
        """Upload detection images and their metadata to cloud storage"""
        try:
            if not os.path.exists(source_dir):
                self.logger.error(f"Source directory does not exist: {source_dir}")
                return False

            # Create a timestamped directory in cloud storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cloud_subdir = os.path.join(self.cloud_dir, f"detections_{timestamp}")
            os.makedirs(cloud_subdir, exist_ok=True)

            # Copy images and metadata
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(cloud_subdir, filename)
                    
                    # Copy image with metadata
                    self._copy_image_with_metadata(source_path, dest_path)

            self.logger.info(f"Successfully uploaded detections to {cloud_subdir}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading detections: {e}")
            return False

    def _copy_image_with_metadata(self, source_path, dest_path):
        """Copy an image and its metadata to the destination"""
        try:
            # Read source image
            with Image.open(source_path) as img:
                # Get EXIF data
                exif_dict = piexif.load(img.info.get('exif', b''))
                
                # Create new EXIF data
                new_exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                
                # Copy existing EXIF data
                for ifd in ("0th", "Exif", "GPS", "1st"):
                    if ifd in exif_dict:
                        new_exif_dict[ifd] = exif_dict[ifd]
                
                # Add upload timestamp
                timestamp = datetime.now().isoformat()
                if 'Exif' in new_exif_dict:
                    new_exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = timestamp.encode()
                
                # Save image with EXIF data
                exif_bytes = piexif.dump(new_exif_dict)
                img.save(dest_path, exif=exif_bytes)
                
            self.logger.info(f"Copied image with metadata: {dest_path}")
        except Exception as e:
            self.logger.error(f"Error copying image {source_path}: {e}")

    def get_image_metadata(self, image_path):
        """Get metadata from an image file"""
        try:
            with Image.open(image_path) as img:
                exif_dict = piexif.load(img.info.get('exif', b''))
                
                metadata = {
                    "filename": os.path.basename(image_path),
                    "size": os.path.getsize(image_path),
                    "modified": os.path.getmtime(image_path),
                    "dimensions": img.size,
                    "format": img.format,
                }
                
                # Extract GPS data if available
                if 'GPS' in exif_dict:
                    gps_data = {}
                    for key in exif_dict['GPS']:
                        if key in piexif.GPSIFD.__dict__:
                            gps_data[piexif.GPSIFD.__dict__[key]] = exif_dict['GPS'][key]
                    metadata['gps'] = gps_data
                
                return metadata
        except Exception as e:
            self.logger.error(f"Error reading metadata from {image_path}: {e}")
            return None

    def list_cloud_contents(self):
        """List all contents in the cloud directory"""
        try:
            contents = []
            for root, dirs, files in os.walk(self.cloud_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.cloud_dir)
                        contents.append({
                            'path': rel_path,
                            'size': os.path.getsize(full_path),
                            'modified': os.path.getmtime(full_path)
                        })
            return contents
        except Exception as e:
            self.logger.error(f"Error listing cloud contents: {e}")
            return []

    def delete_image(self, image_path):
        """Delete an image from cloud storage"""
        try:
            full_path = os.path.join(self.cloud_dir, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                self.logger.info(f"Deleted image: {full_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting image {image_path}: {e}")
            return False

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
        """Initialize cloud storage client using credentials from .env"""
        try:
            # Check if required settings are present
            required_env_vars = [
                'GOOGLE_TYPE',
                'GOOGLE_PROJECT_ID',
                'GOOGLE_PRIVATE_KEY_ID',
                'GOOGLE_PRIVATE_KEY',
                'GOOGLE_CLIENT_EMAIL',
                'GOOGLE_CLIENT_ID',
                'GOOGLE_AUTH_URI',
                'GOOGLE_TOKEN_URI',
                'GOOGLE_AUTH_PROVIDER_CERT_URL',
                'GOOGLE_CLIENT_CERT_URL',
                'GOOGLE_UNIVERSE_DOMAIN'
            ]
            
            # Check if all required environment variables are present
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                self.logger.warning(f"Cloud storage not initialized - missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            # Create credentials dictionary from environment variables
            credentials_dict = {
                "type": os.getenv('GOOGLE_TYPE'),
                "project_id": os.getenv('GOOGLE_PROJECT_ID'),
                "private_key_id": os.getenv('GOOGLE_PRIVATE_KEY_ID'),
                "private_key": os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
                "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                "auth_uri": os.getenv('GOOGLE_AUTH_URI'),
                "token_uri": os.getenv('GOOGLE_TOKEN_URI'),
                "auth_provider_x509_cert_url": os.getenv('GOOGLE_AUTH_PROVIDER_CERT_URL'),
                "client_x509_cert_url": os.getenv('GOOGLE_CLIENT_CERT_URL'),
                "universe_domain": os.getenv('GOOGLE_UNIVERSE_DOMAIN')
            }
            
            # Initialize client with credentials from environment
            self.client = storage.Client.from_service_account_info(credentials_dict)
            
            # Get bucket
            if not self.settings['bucket_name']:
                self.logger.warning("Cloud storage not initialized - missing bucket name")
                return False
                
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
            
    def upload_detection(self, image: Any, defect_counts: Dict[str, int], frame_counts: Dict[str, int], bboxes: Optional[list] = None, metadata: Optional[Dict] = None, original_filename: Optional[str] = None) -> bool:
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

            # Create blob names with folder path and original filename if provided
            folder_path = self.settings['folder_path'].rstrip('/')
            if original_filename:
                # Use original filename but ensure it's unique with timestamp
                base_name = os.path.splitext(original_filename)[0]
                image_blob_name = f"{folder_path}/{base_name}_{timestamp}.jpg"
            else:
                image_blob_name = f"{folder_path}/detection_{timestamp}.jpg"

            # Prepare metadata (convert numpy types to Python types)
            def to_python_type(val):
                if hasattr(val, "item"):
                    return val.item()
                if isinstance(val, np.ndarray):
                    return val.tolist()
                return val

            # Combine provided metadata with detection data
            upload_metadata = {
                'timestamp': timestamp,
                'defect_counts': {k: to_python_type(v) for k, v in defect_counts.items()},
                'frame_counts': {k: to_python_type(v) for k, v in frame_counts.items()},
                'project_id': self.settings['project_id'],
                'region': self.settings['region'],
                'bucket': self.settings['bucket_name'],
                'folder_path': folder_path
            }

            # Add additional metadata if provided
            if metadata:
                # Convert any numpy types in metadata
                processed_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        processed_metadata[k] = to_python_type(v)
                    else:
                        processed_metadata[k] = v
                upload_metadata.update(processed_metadata)

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
            user_comment = json.dumps(upload_metadata)
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

            # Also upload metadata as a separate JSON file
            metadata_blob_name = os.path.splitext(image_blob_name)[0] + '_metadata.json'
            metadata_blob = self.bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_string(
                json.dumps(upload_metadata, indent=4),
                content_type='application/json'
            )

            self.logger.info(f"Uploaded detection image and metadata {timestamp} to cloud storage")
            self.logger.info(f"Image: {image_blob_name}")
            self.logger.info(f"Metadata: {metadata_blob_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error uploading detection: {e}")
            return False
            
    def list_detections(self) -> List[str]:
        """List all detection files in the cloud storage"""
        if not self.is_initialized or not self.bucket:
            return []
            
        try:
            # List all blobs in the bucket with the specified folder path
            prefix = self.settings['folder_path'] if self.settings['folder_path'] else ''
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            # Filter for image files
            image_files = []
            for blob in blobs:
                if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(blob.name)
                    
            return image_files
        except Exception as e:
            self.logger.error(f"Error listing detections: {e}")
            return []

    def download_detection(self, blob_name: str) -> Tuple[Optional[bytes], Optional[Dict]]:
        """Download a detection image and its metadata from cloud storage"""
        if not self.is_initialized or not self.bucket:
            return None, None
            
        try:
            # Get the blob
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                self.logger.error(f"Blob {blob_name} does not exist")
                return None, None
                
            # Download the image data
            image_data = blob.download_as_bytes()
            
            # Try to get metadata from blob metadata
            metadata = blob.metadata if blob.metadata else {}
            
            # Add basic file information
            metadata.update({
                'filename': os.path.basename(blob_name),
                'size': blob.size,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type
            })
            
            return image_data, metadata
            
        except Exception as e:
            self.logger.error(f"Error downloading detection {blob_name}: {e}")
            return None, None

    def delete_detection(self, blob_name: str) -> bool:
        """Delete a detection from cloud storage"""
        if not self.is_initialized or not self.bucket:
            return False
            
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                self.logger.info(f"Deleted detection: {blob_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting detection {blob_name}: {e}")
            return False

    def upload_detection(self, image_data: bytes, metadata: Dict[str, Any], filename: str) -> bool:
        """Upload a detection image with metadata to cloud storage"""
        if not self.is_initialized or not self.bucket:
            return False
            
        try:
            # Create blob name with folder path if specified
            blob_name = filename
            if self.settings['folder_path']:
                blob_name = os.path.join(self.settings['folder_path'], filename)
                
            # Create blob and upload
            blob = self.bucket.blob(blob_name)
            blob.metadata = metadata
            blob.upload_from_string(
                image_data,
                content_type='image/jpeg'  # or determine from filename
            )
            
            self.logger.info(f"Uploaded detection: {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading detection {filename}: {e}")
            return False

    def download_metadata(self, blob_name: str) -> Optional[Dict]:
        """Download metadata from a JSON file in cloud storage"""
        if not self.is_initialized or not self.bucket:
            return None
            
        try:
            # Get the blob
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                self.logger.error(f"Metadata blob {blob_name} does not exist")
                return None
                
            # Download and parse JSON data
            json_data = blob.download_as_string()
            metadata = json.loads(json_data)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error downloading metadata {blob_name}: {e}")
            return None

    def upload_metadata(self, metadata: Dict[str, Any], filename: str) -> bool:
        """Upload metadata as JSON to cloud storage"""
        if not self.is_initialized or not self.bucket:
            return False
            
        try:
            # Create blob name with folder path if specified
            blob_name = filename
            if self.settings['folder_path']:
                blob_name = os.path.join(self.settings['folder_path'], filename)
                
            # Create blob and upload JSON
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )
            
            self.logger.info(f"Uploaded metadata: {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading metadata {filename}: {e}")
            return False

    def __del__(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")