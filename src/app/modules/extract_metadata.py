import os
import json
import re
from PIL import Image
from typing import List, Tuple
import logging
import piexif
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_enhanced_metadata(image_path: str) -> Tuple[List[Tuple[int, int, int, int]], Tuple[float, float]]:
    """Extract defect bounding boxes and location from EXIF UserComment metadata using piexif."""
    bboxes = []
    location = (None, None)  # Initialize location
    try:
        img = Image.open(image_path)
        exif_data = img.info.get("exif", None)

        if not exif_data:
            logger.info("No EXIF metadata found.")
            return bboxes, location

        exif_dict = piexif.load(exif_data)
        user_comment_bytes = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, None)

        if user_comment_bytes:
            # Decode EXIF-compliant UserComment
            try:
                user_comment = user_comment_bytes
                if user_comment_bytes.startswith(b"ASCII\0\0\0"):
                    user_comment = user_comment_bytes[8:].decode('utf-8')
                else:
                    user_comment = user_comment_bytes.decode('utf-8', errors='ignore')

                logger.info("🔍 Retrieved Metadata:\n" + user_comment)

                # Parse the JSON string
                defect_data = json.loads(user_comment)
                defects = defect_data.get("Defects", [])
                location = defect_data.get("Location", (None, None))  # Extract location

                for defect in defects:
                    bbox = defect.get("bbox", None)
                    if bbox and len(bbox) == 4:
                        bboxes.append(tuple(bbox))
            except Exception as e:
                logger.error(f"Failed to decode or parse UserComment: {e}")
        else:
            logger.info("No UserComment found in metadata.")
    except Exception as e:
        logger.error(f"Error reading metadata: {e}")
    return bboxes, location  # Return both bounding boxes and location

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
    except Exception as e:
        logger.error(f"Error during extraction: {e}")