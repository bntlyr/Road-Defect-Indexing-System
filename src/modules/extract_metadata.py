import os
import json
import re
from PIL import Image
from typing import List, Tuple, Optional
import logging
import piexif

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_image_metadata(image_path: str) -> Tuple[Dict, Optional[Tuple[float, float]], List[Tuple[int, int, int, int]]]:
    """
    Extract metadata, GPS information, and bounding boxes from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing (metadata_dict, gps_location, bounding_boxes)
    """
    try:
        # Load the image and extract EXIF data
        img = exif.Image(image_path)
        metadata = {tag: str(getattr(img, tag)) for tag in img.list_all()}

        gps_location = None
        # Extract GPS data if available
        if img.has_exif and img.gps_latitude and img.gps_longitude:
            lat = img.gps_latitude
            lon = img.gps_longitude
            lat_ref = img.gps_latitude_ref
            lon_ref = img.gps_longitude_ref
            
            # Convert to decimal degrees
            latitude = lat[0] + lat[1] / 60 + lat[2] / 3600
            longitude = lon[0] + lon[1] / 60 + lon[2] / 3600
            
            if lat_ref != 'N':
                latitude = -latitude
            if lon_ref != 'E':
                longitude = -longitude
            
            gps_location = (latitude, longitude)

        return metadata, gps_location, []
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {}, None, []