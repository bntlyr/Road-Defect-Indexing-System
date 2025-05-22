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

def _convert_from_exif_gps(gps_coords: tuple) -> Optional[float]:
    """Convert EXIF GPS coordinate format to decimal degrees."""
    try:
        if not isinstance(gps_coords, tuple) or len(gps_coords) != 3:
            return None
            
        degrees, minutes, seconds = gps_coords
        if not all(isinstance(x, tuple) and len(x) == 2 for x in [degrees, minutes, seconds]):
            return None
            
        deg_val = degrees[0] / degrees[1]
        min_val = minutes[0] / minutes[1]
        sec_val = seconds[0] / seconds[1]
        
        return deg_val + (min_val / 60.0) + (sec_val / 3600.0)
    except Exception as e:
        logger.error(f"Error converting EXIF GPS coordinate: {e}")
        return None

def _get_location_from_exif(exif_dict: dict) -> Optional[Tuple[float, float]]:
    """Extract location from EXIF GPS data."""
    try:
        if 'GPS' not in exif_dict:
            return None
            
        gps = exif_dict['GPS']
        required_tags = [
            piexif.GPSIFD.GPSLatitudeRef,
            piexif.GPSIFD.GPSLatitude,
            piexif.GPSIFD.GPSLongitudeRef,
            piexif.GPSIFD.GPSLongitude
        ]
        
        if not all(tag in gps for tag in required_tags):
            return None
            
        lat = _convert_from_exif_gps(gps[piexif.GPSIFD.GPSLatitude])
        lon = _convert_from_exif_gps(gps[piexif.GPSIFD.GPSLongitude])
        
        if lat is None or lon is None:
            return None
            
        # Apply N/S and E/W signs
        if gps[piexif.GPSIFD.GPSLatitudeRef] == b'S':
            lat = -lat
        if gps[piexif.GPSIFD.GPSLongitudeRef] == b'W':
            lon = -lon
            
        # Validate coordinates
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
            
        logger.warning(f"Invalid GPS coordinates from EXIF: {lat}, {lon}")
        return None
    except Exception as e:
        logger.error(f"Error extracting location from EXIF: {e}")
        return None

def _get_location_from_usercomment(user_comment: str) -> Optional[Tuple[float, float]]:
    """Extract location from UserComment JSON data."""
    try:
        defect_data = json.loads(user_comment)
        location = defect_data.get("Location", (None, None))
        
        if not isinstance(location, tuple) or len(location) != 2:
            return None
            
        lat, lon = location
        if lat is None or lon is None:
            return None
            
        # Validate coordinates
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
            
        logger.warning(f"Invalid GPS coordinates from UserComment: {lat}, {lon}")
        return None
    except Exception as e:
        logger.error(f"Error extracting location from UserComment: {e}")
        return None

def read_enhanced_metadata(image_path: str) -> Tuple[List[Tuple[int, int, int, int]], Tuple[float, float]]:
    """Extract defect bounding boxes and location from both EXIF GPS and UserComment metadata."""
    bboxes = []
    location = (None, None)  # Initialize location
    
    try:
        img = Image.open(image_path)
        exif_data = img.info.get("exif", None)

        if not exif_data:
            logger.info("No EXIF metadata found.")
            return bboxes, location

        exif_dict = piexif.load(exif_data)
        
        # Try to get location from EXIF GPS data first
        location = _get_location_from_exif(exif_dict)
        if location:
            logger.info(f"Location found in EXIF GPS data: {location}")
        
        # Try to get location from UserComment if not found in EXIF
        user_comment_bytes = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, None)
        if user_comment_bytes and location == (None, None):
            try:
                # Decode EXIF-compliant UserComment
                user_comment = user_comment_bytes
                if user_comment_bytes.startswith(b"ASCII\0\0\0"):
                    user_comment = user_comment_bytes[8:].decode('utf-8')
                else:
                    user_comment = user_comment_bytes.decode('utf-8', errors='ignore')

                logger.info("🔍 Retrieved Metadata:\n" + user_comment)
                
                # Try to get location from UserComment
                usercomment_location = _get_location_from_usercomment(user_comment)
                if usercomment_location:
                    location = usercomment_location
                    logger.info(f"Location found in UserComment: {location}")
                
                # Extract bounding boxes
                defect_data = json.loads(user_comment)
                defects = defect_data.get("Defects", [])
                
                for defect in defects:
                    bbox = defect.get("bbox", None)
                    if bbox and len(bbox) == 4:
                        bboxes.append(tuple(bbox))
                        
            except Exception as e:
                logger.error(f"Failed to decode or parse UserComment: {e}")
        elif not user_comment_bytes:
            logger.info("No UserComment found in metadata.")
            
    except Exception as e:
        logger.error(f"Error reading metadata: {e}")
        
    return bboxes, location

if __name__ == "__main__":
    # Hardcoded image path for testing
    image_path = r"C:/Users/bentl/RoadDefectDetections/detection_20250516_205938.jpg"

    try:
        bboxes, location = read_enhanced_metadata(image_path)
        if bboxes:
            logger.info("Extracted Bounding Boxes:")
            for bbox in bboxes:
                logger.info(bbox)
        else:
            logger.info("No defect bounding boxes found in the image.")
            
        if location != (None, None):
            logger.info(f"Extracted Location: {location}")
        else:
            logger.info("No location data found in the image.")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")