from PIL import Image as PILImage
import piexif
import piexif.helper
import io
import json

def extract_metadata_from_image(image_bytes: bytes) -> dict:
    try:
        # Load the image from bytes
        img = PILImage.open(io.BytesIO(image_bytes))

        # Try to get EXIF from _getexif() if not present in info
        exif_data = img.info.get("exif", None)
        if not exif_data and hasattr(img, "_getexif"):
            raw_exif = img._getexif()
            if raw_exif:
                # Convert to bytes for piexif if possible
                try:
                    exif_data = piexif.dump(raw_exif)
                except Exception:
                    exif_data = None

        if not exif_data:
            print("No EXIF data found.")
            return {}

        # Load EXIF dict from bytes
        exif_dict = piexif.load(exif_data)

        # Get the UserComment field
        user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment, None)
        if user_comment:
            # Decode using piexif helper (handles ASCII prefix)
            decoded_comment = piexif.helper.UserComment.load(user_comment)
            # Parse the JSON metadata
            metadata = json.loads(decoded_comment)
            return metadata
        else:
            print("No UserComment field found in EXIF.")
            return {}

    except Exception as e:
        print(f"Failed to extract metadata: {e}")
        return {}

if __name__ == "__main__":
    image_path = "./output/processed_road.jpg"
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    metadata = extract_metadata_from_image(image_bytes)
    
    for key, value in metadata.items():
        print(f"{key}: {value}")
