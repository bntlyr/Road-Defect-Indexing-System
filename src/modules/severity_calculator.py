import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import os
import json
from datetime import datetime
from PIL import Image
import logging
from skimage import exposure
from skimage.filters import threshold_niblack
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from . import fuzzy_logic
from . import random_forest
from . import traffic_getter
import torch
from ultralytics import YOLOv10 as YOLO
import piexif
from src.modules.fuzzy_logic import calculate_severity_percentage
from src.modules.random_forest import predict_repair_probability, MODEL_AVAILABLE
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using multiple techniques."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE with stronger parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply additional contrast stretching
    p2, p98 = np.percentile(enhanced, (2, 98))
    enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))
    
    # Apply gamma correction
    gamma = 1.2  # Slightly increase contrast
    enhanced = np.power(enhanced / 255.0, 1.0/gamma) * 255.0
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced

def apply_canny_edge_detection(image: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection to the image."""
    return cv2.Canny(image, 100, 200)

def apply_superpixel_segmentation(image_rgb: np.ndarray, num_segments: int = 500, compactness: int = 5) -> np.ndarray:
    """Apply SLIC superpixel segmentation."""
    return slic(image_rgb, n_segments=num_segments, compactness=compactness, start_label=1)

def overlay_color_mask(base_image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    """Overlay a color mask on the base image."""
    color_mask = np.zeros_like(base_image)
    color_mask[mask > 0] = color
    return cv2.addWeighted(base_image, 1 - alpha, color_mask, alpha, 0)

def nlm_denoising(image: np.ndarray) -> np.ndarray:
    """Apply Non-local Means Denoising with moderate parameters."""
    # Ensure input is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Apply moderate NLM denoising
    denoised = cv2.fastNlMeansDenoising(
        image, 
        None,
        h=12,  # Reduced from 15 to 12 for more moderate denoising
        templateWindowSize=7,  # Reduced from 9 to 7
        searchWindowSize=21    # Reduced from 25 to 21
    )
    return denoised

def draw_bboxes(image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on the image."""
    result = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    return result

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
        exif_data = piexif.load(image_path)
        metadata = {}
        
        # Extract UserComment from EXIF data
        user_comment = exif_data['Exif'].get(0x9286, None)  # 0x9286 is the tag for UserComment
        gps_location = None
        
        if user_comment:
            logger.info(f"UserComment: {user_comment}")
            try:
                # Decode the UserComment if it's in bytes and remove the ASCII prefix
                user_comment = user_comment.decode('utf-8')
                # Remove the ASCII prefix if present
                if user_comment.startswith('ASCII\x00'):
                    user_comment = user_comment[8:]  # Remove the prefix

                # Parse UserComment as JSON
                location_data = json.loads(user_comment)
                
                # Try to get GPS location from different possible formats
                if "detections" in location_data:
                    detections = location_data["detections"]
                    
                    # Try GPS array format first
                    if "GPS" in detections and isinstance(detections["GPS"], list) and len(detections["GPS"]) >= 2:
                        gps_location = (float(detections["GPS"][0]), float(detections["GPS"][1]))
                        logger.info(f"Found GPS location from GPS array: {gps_location}")
                    
                    # Try Location string format if GPS array wasn't found
                    elif "Location" in detections:
                        location_str = detections["Location"]
                        if isinstance(location_str, str):
                            # Split the string and convert to floats
                            lat_str, lon_str = location_str.split(",")
                            gps_location = (float(lat_str.strip()), float(lon_str.strip()))
                            logger.info(f"Found GPS location from Location string: {gps_location}")
                
                # Try new metadata format with Location object
                elif "Location" in location_data and isinstance(location_data["Location"], dict):
                    location = location_data["Location"]
                    if "latitude" in location and "longitude" in location:
                        gps_location = (float(location["latitude"]), float(location["longitude"]))
                        logger.info(f"Found GPS location from Location object: {gps_location}")
                
                # Store the full metadata
                metadata = location_data
                
            except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
                logger.warning(f"Error parsing UserComment for location: {e}")

        return metadata, gps_location, []  # Return empty list for bounding boxes
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {}, None, []  # Return empty list for bounding boxes


def apply_pinhole_camera_model(image: np.ndarray, camera_matrix: np.ndarray, 
                             distortion_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply camera calibration to undistort the image.
    
    Args:
        image: Input image
        camera_matrix: Camera calibration matrix
        distortion_coeffs: Distortion coefficients
        
    Returns:
        Undistorted image
    """
    try:
        return cv2.undistort(image, camera_matrix, distortion_coeffs)
    except Exception as e:
        logger.error(f"Error in camera undistortion: {e}")
        return image

def apply_canny_edge_detection(contrast_img: np.ndarray, 
                             low_thresh: int = 50, 
                             high_thresh: int = 150) -> np.ndarray:
    """Apply Canny edge detection to the image."""
    return cv2.Canny(contrast_img, low_thresh, high_thresh)

def apply_superpixel_segmentation(image: np.ndarray, 
                                n_segments: int = 100,
                                compactness: float = 10.0) -> np.ndarray:
    """Apply SLIC superpixel segmentation."""
    return slic(image, n_segments=n_segments, compactness=compactness, 
               start_label=1, channel_axis=2)

def overlay_segmentation_on_image(image: np.ndarray, 
                                segments: np.ndarray,
                                color: Tuple[int, int, int] = (255, 0, 0),
                                alpha: float = 0.2) -> np.ndarray:
    """Overlay segmentation boundaries on the image."""
    boundaries = np.zeros_like(image)
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        boundaries[mask] = color
    
    return cv2.addWeighted(image, 1, boundaries, alpha, 0)

def adjust_bboxes_after_undistortion(image: np.ndarray, camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Adjust bounding boxes after image undistortion."""
    h, w = image.shape[:2]
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_mtx)
    transformed_bboxes = []
    for (x1, y1, x2, y2) in bboxes:
        points = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        undistorted_points = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=new_camera_mtx)
        coords = undistorted_points.reshape(-1, 2)
        min_x, min_y = np.min(coords[:, 0]), np.min(coords[:, 1])
        max_x, max_y = np.max(coords[:, 0]), np.max(coords[:, 1])
        # Clamp to image dimensions
        min_x = max(0, min(int(round(min_x)), w - 1))
        max_x = max(0, min(int(round(max_x)), w - 1))
        min_y = max(0, min(int(round(min_y)), h - 1))
        max_y = max(0, min(int(round(max_y)), h - 1))
        transformed_bboxes.append((min_x, min_y, max_x, max_y))
    return undistorted_image, transformed_bboxes

class SeverityCalculator:
    """
    Calculate the severity of road defects based on image analysis.
    """
    
    def __init__(self, camera_width: int, camera_height: int, 
                 focal_length: float = 35.0,
                 sensor_width: float = 36.0,
                 sensor_height: float = 24.0,
                 model_path: Optional[str] = None):
        """
        Initialize the severity calculator.
        
        Args:
            camera_width: Camera sensor width in pixels
            camera_height: Camera sensor height in pixels
            focal_length: Camera focal length in mm
            sensor_width: Physical sensor width in mm
            sensor_height: Physical sensor height in mm
            model_path: Path to YOLO model for defect detection
        """
        if camera_width <= 0 or camera_height <= 0:
            raise ValueError("Camera dimensions must be positive")
            
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        
        # Calculate pixel size and field of view
        self.pixel_size_x = sensor_width / camera_width
        self.pixel_size_y = sensor_height / camera_height
        
        if focal_length > 0:
            self.fov_x = 2 * np.arctan(sensor_width / (2 * focal_length))
            self.fov_y = 2 * np.arctan(sensor_height / (2 * focal_length))
        else:
            self.fov_x = self.fov_y = 0
            logger.warning("Invalid focal length provided")

        # Initialize YOLO model if path is provided
        self.model = None
        if model_path:
            try:
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at: {model_path}")
                else:
                    logger.info(f"Loading YOLO model from {model_path}")
                    self.model = YOLO(model_path)
                    # Verify model loaded correctly by running a test prediction
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    test_results = self.model(test_img, conf=0.15)[0]
                    logger.info("YOLO model loaded and verified successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                logger.exception("Full traceback:")
        else:
            logger.warning("No YOLO model path provided")

    def detect_defects(self, image: np.ndarray, confidence_threshold: float = 0.20) -> List[Dict]:
        """
        Detect defects in the image using YOLO model.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence score for detections (default: 0.15)
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        if self.model is None:
            logger.warning("No YOLO model available for defect detection")
            return []

        try:
            # Ensure image is in the correct format
            if len(image.shape) == 2:  # If grayscale, convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Resize image to 640x640 while maintaining aspect ratio
            h, w = image.shape[:2]
            scale = min(640 / w, 640 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(image, (new_w, new_h))
            
            # Create a square canvas of 640x640
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            # Place the resized image in the center
            y_offset = (640 - new_h) // 2
            x_offset = (640 - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
            
            # Run inference with fixed size
            results = self.model(
                canvas,
                conf=confidence_threshold,
                imgsz=640,  # Use fixed size of 640x640
                verbose=False,
                agnostic_nms=True,
                max_det=50
            )[0]
            
            detections = []
            
            # Process results and scale back to original image size
            for box in results.boxes:
                # Get coordinates in 640x640 space
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                
                # Remove padding and scale back to original image size
                x1 = (x1 - x_offset) / scale
                y1 = (y1 - y_offset) / scale
                x2 = (x2 - x_offset) / scale
                y2 = (y2 - y_offset) / scale
                
                # Convert to integers and ensure within bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Log each detection for debugging
                logger.info(f"Detection: {class_name} at ({x1}, {y1}, {x2}, {y2}) with confidence {confidence:.3f}")
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class': class_name
                })
            
            logger.info(f"Detected {len(detections)} defects with confidence threshold {confidence_threshold}")
            return detections
            
        except Exception as e:
            logger.error(f"Error during defect detection: {e}")
            logger.exception("Full traceback:")
            return []

    def calculate_defect_area(self, 
                            image_roi_bgr: np.ndarray,
                            bbox_relative_to_roi: Tuple[int, int, int, int] = (0,0,-1,-1),
                            full_image_size: Optional[Tuple[int, int]] = None
                            ) -> Tuple[int, int, float]:
        """
        Calculate defect area within the provided image ROI, relative to the full image size.
        
        Args:
            image_roi_bgr: Input image in BGR format
            bbox_relative_to_roi: Bounding box coordinates (x1,y1,x2,y2)
            full_image_size: Optional tuple of (width, height) of the full image
            
        Returns:
            Tuple of (defect_area_pixels, total_processed_area_pixels, relative_area_ratio)
        """
        if image_roi_bgr is None or image_roi_bgr.size == 0:
            return 0, 0, 0.0

        # Convert to grayscale if needed
        if len(image_roi_bgr.shape) == 3:
            gray_roi = cv2.cvtColor(image_roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = image_roi_bgr
        
        # Process ROI based on bounding box
        if bbox_relative_to_roi[2] != -1:
            x1, y1, x2, y2 = bbox_relative_to_roi
            gray_roi_processed = gray_roi[y1:y2, x1:x2].copy()
        else:
            gray_roi_processed = gray_roi.copy()

        if gray_roi_processed.size == 0:
            return 0, 0, 0.0

        # Apply thresholding and noise reduction
        var_mask = self._apply_variance_thresholding(gray_roi_processed)
        nib_mask = self._apply_niblack_thresholding(gray_roi_processed)
        
        combined_mask = cv2.bitwise_or(var_mask, nib_mask)
        clean_mask = self._reduce_noise(combined_mask)
        
        defect_area_pixels = np.sum(clean_mask == 255)
        total_processed_area_pixels = gray_roi_processed.size

        # Calculate relative area ratio if full image size is provided
        relative_area_ratio = 0.0
        if full_image_size is not None:
            full_image_area = full_image_size[0] * full_image_size[1]
            relative_area_ratio = defect_area_pixels / full_image_area if full_image_area > 0 else 0.0
        
        return defect_area_pixels, total_processed_area_pixels, relative_area_ratio

    def _apply_variance_thresholding(self, gray_roi: np.ndarray, 
                                   var_thresh_val: int = 12) -> np.ndarray:  # Reduced threshold
        """Apply variance-based thresholding with moderate parameters."""
        if gray_roi.size == 0:
            return np.array([], dtype=np.uint8)
        
        # Apply Otsu's thresholding with light preprocessing
        blur = cv2.GaussianBlur(gray_roi, (3,3), 0)  # Reduced kernel size
        _, thresh_img = cv2.threshold(blur, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate local variance
        variance = np.var(gray_roi)
        
        if variance < var_thresh_val:
            return np.zeros_like(gray_roi, dtype=np.uint8)
        
        # Apply light morphological operations
        kernel = np.ones((2,2), np.uint8)  # Reduced kernel size
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        
        return thresh_img

    def _apply_niblack_thresholding(self, gray_roi: np.ndarray,
                                  window_size: int = 35,  # Reduced window size
                                  k: float = 0.12) -> np.ndarray:  # Adjusted k value
        """Apply Niblack thresholding with moderate parameters."""
        if gray_roi.size == 0:
            return np.array([], dtype=np.uint8)
        
        min_dim = min(gray_roi.shape)
        if min_dim < window_size:
            window_size = max(3, min_dim if min_dim % 2 != 0 else min_dim - 1)
        
        # Apply light preprocessing
        blur = cv2.GaussianBlur(gray_roi, (3,3), 0)  # Reduced kernel size
        
        # Apply Niblack thresholding
        thresh_values = threshold_niblack(blur, window_size=window_size, k=k)
        binary_img = blur <= thresh_values
        
        # Apply light morphological operations
        kernel = np.ones((2,2), np.uint8)  # Reduced kernel size
        binary_img = cv2.morphologyEx((binary_img * 255).astype(np.uint8), 
                                    cv2.MORPH_CLOSE, kernel)
        
        return binary_img

    def _reduce_noise(self, binary_mask: np.ndarray,
                     kernel_size: int = 3,
                     min_size: int = 50) -> np.ndarray:
        """Reduce noise in binary mask using morphological operations."""
        if binary_mask.size == 0 or binary_mask.dtype != np.uint8:
            return np.array([], dtype=np.uint8)
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        output_mask = np.zeros_like(closed)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output_mask[labels == i] = 255
                
        return output_mask

    def calculate_real_world_size(self, pixel_area: int, 
                                distance_to_object_m: float) -> float:
        """
        Calculate real-world size of defect in square meters.
        
        Args:
            pixel_area: Area in pixels
            distance_to_object_m: Distance to object in meters
            
        Returns:
            Real-world area in square meters
        """
        if self.focal_length == 0:
            return 0.0
            
        sensor_area_mm2 = pixel_area * self.pixel_size_x * self.pixel_size_y
        real_world_area_mm2 = (sensor_area_mm2 * (distance_to_object_m * 1000)**2) / (self.focal_length**2)
        return real_world_area_mm2 / (1000**2)

    def calculate_severity(self, 
                         image: np.ndarray,
                         detections: List[Dict],
                         distance_to_object_m: float = 1.0
                         ) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate severity metrics for detected defects relative to the whole image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries with 'bbox' key
            distance_to_object_m: Distance to object in meters
            
        Returns:
            Tuple of (severity_level, real_world_area_m2, total_bbox_area, 
                     avg_length_cm, avg_width_cm, defect_ratio)
        """
        if not detections or image is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        total_defect_pixels = 0
        total_bbox_pixels = 0
        total_length_cm = 0.0
        total_width_cm = 0.0
        detection_count = 0

        # Get full image dimensions
        full_image_height, full_image_width = image.shape[:2]
        full_image_size = (full_image_width, full_image_height)
        total_image_pixels = full_image_width * full_image_height

        # Process each detection
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0:
                continue
                
            # Ensure ROI coordinates are within bounds
            roi_x1, roi_y1 = max(0, x1), max(0, y1)
            roi_x2, roi_y2 = min(full_image_width, x2), min(full_image_height, y2)
            
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                continue
                
            # Extract ROI for processing
            current_roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            bbox_relative = (0, 0, roi_x2 - roi_x1, roi_y2 - roi_y1)
            
            # Calculate defect area relative to full image
            defect_pixels, roi_pixels, relative_area = self.calculate_defect_area(
                current_roi, 
                bbox_relative,
                full_image_size
            )
            
            # Calculate real-world dimensions using camera parameters
            if self.focal_length > 0:
                # Convert pixel dimensions to real-world measurements (in cm)
                # Use the relative area to scale the measurements
                length_cm = (w * self.pixel_size_x * distance_to_object_m * 100) / self.focal_length
                width_cm = (h * self.pixel_size_y * distance_to_object_m * 100) / self.focal_length
                
                # Scale measurements based on relative area
                length_cm *= np.sqrt(relative_area) if relative_area > 0 else 1.0
                width_cm *= np.sqrt(relative_area) if relative_area > 0 else 1.0
                
                total_length_cm += length_cm
                total_width_cm += width_cm
                detection_count += 1
            
            total_defect_pixels += defect_pixels
            total_bbox_pixels += roi_pixels

        # Calculate averages
        avg_length_cm = total_length_cm / detection_count if detection_count > 0 else 0.0
        avg_width_cm = total_width_cm / detection_count if detection_count > 0 else 0.0
        
        # Calculate severity metrics relative to full image
        defect_ratio = total_defect_pixels / total_image_pixels if total_image_pixels > 0 else 0.0
        
        # Calculate real-world area using the total defect pixels relative to full image
        real_world_area = self.calculate_real_world_size(
            int(total_defect_pixels * (total_image_pixels / total_bbox_pixels if total_bbox_pixels > 0 else 1.0)),
            distance_to_object_m
        )
        
        # Calculate severity level based on defect ratio relative to full image
        severity_level = min(1.0, defect_ratio * 2.0)
        
        return severity_level, real_world_area, total_bbox_pixels, avg_length_cm, avg_width_cm, defect_ratio



    def enhance_crack_visibility(self, image: np.ndarray) -> np.ndarray:
        """Enhanced crack visibility using image preprocessing and superpixel segmentation."""
        # Step 1: Grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 3: Contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=0)

        # Step 4: Median filtering
        median = cv2.medianBlur(enhanced, 3)

        # Step 5: Adaptive thresholding (Otsu fallback)
        try:
            # Try adaptive thresholding first
            morph = cv2.adaptiveThreshold(
                median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        except Exception:
            # Fallback to Otsu if adaptive fails
            _, morph = cv2.threshold(
                median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Step 6: Morphological filtering (open → close)
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Step 7: Superpixel segmentation overlay
        superpixel_mask = apply_superpixel_segmentation(image, n_segments=200, compactness=15.0)

        # Resize and convert to grayscale if needed
        if superpixel_mask.shape[:2] != morph.shape:
            superpixel_mask = cv2.resize(superpixel_mask, (morph.shape[1], morph.shape[0]), interpolation=cv2.INTER_NEAREST)

        if len(superpixel_mask.shape) == 3:
            superpixel_mask = cv2.cvtColor(superpixel_mask, cv2.COLOR_RGB2GRAY)

        # Normalize and convert to uint8
        superpixel_mask = cv2.normalize(superpixel_mask, None, 0, 255, cv2.NORM_MINMAX)
        superpixel_mask = superpixel_mask.astype(np.uint8)

        # Combine cracks with superpixel map
        superpixel_overlay = cv2.bitwise_and(superpixel_mask, morph)

        # Final smoothing (light blur to reduce pixel-level noise)
        cleaned = cv2.medianBlur(superpixel_overlay, 3)

        return cleaned




    def process_image(self,
                     image_path: str,
                     camera_matrix: np.ndarray,
                     distortion_coeffs: np.ndarray,
                     save_path: str,
                     distance_to_object_m: float = 1.0,
                     confidence_threshold: float = 0.15,
                     auto_delete_raw_failed_detections = False) -> Tuple[Optional[np.ndarray], List[int], float, Dict]:
                     
        """
        Process an image to detect and analyze road defects using the improved pipeline.
        """
        try:
            # Extract metadata and GPS (ignore bounding boxes from metadata)
            metadata, gps_loc, user_comment = extract_image_metadata(image_path)
            logger.info(f"Processing image: {image_path}")
            logger.info(f"GPS location: {gps_loc}")
            
            # Load and undistort image
            original_img = cv2.imread(image_path)
            if original_img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None, [], 0.0, {}
            
            # Get optimal camera matrix and undistort
            h, w = original_img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
            undistorted_img = cv2.undistort(original_img, camera_matrix, distortion_coeffs, None, new_camera_mtx)
            
            # Run YOLO inference on the original undistorted image
            detections = self.detect_defects(undistorted_img, confidence_threshold)
            logger.info(f"YOLO detected {len(detections)} defects")
            
            # Skip processing if no defects are detected
            if not detections:
                logger.warning(f"No defects detected in image: {image_path}. Skipping...")
                if auto_delete_raw_failed_detections:
                    try:
                        os.remove(image_path)
                        logger.info(f"Deleted raw image: {image_path} due to no detections")
                    except Exception as e:
                        logger.error(f"Failed to delete image {image_path}: {e}")
                return None, [], 0.0, {}
            
            # Create enhanced crack visualization
            enhanced_cracks = self.enhance_crack_visibility(undistorted_img)
            
            # Run YOLO inference on the original undistorted image first
            detections = self.detect_defects(undistorted_img, confidence_threshold)
            logger.info(f"YOLO detected {len(detections)} defects")
            
            # If no defects detected, try with enhanced image
            if not detections:
                logger.info("No defects detected in original image, trying with enhanced image...")
                enhanced_bgr = cv2.cvtColor(enhanced_cracks, cv2.COLOR_GRAY2BGR)
                detections = self.detect_defects(enhanced_bgr, confidence_threshold)
                logger.info(f"YOLO detected {len(detections)} defects after enhancement")
            
            # Initialize visualization and metrics
            visualization = undistorted_img.copy()
            total_defect_pixels = 0
            total_bbox_pixels = 0
            
            # Define colors for different defect types with higher saturation
            defect_colors = {
                'Linear-Crack': (0, 255, 0),     # Bright Green
                'Alligator-Crack': (0, 255, 0),  # Bright Orange
                'pothole': (0, 255, 0)           # Bright Red
            }
            
            # Process each detection with enhanced visualization
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class']
                
                # Get color for this defect type
                color = defect_colors.get(class_name, (0, 255, 0))
                
                # Extract ROI for this detection
                x1, y1, x2, y2 = bbox
                roi = enhanced_cracks[y1:y2, x1:x2]
                
                # Create a mask for this ROI
                roi_mask = np.zeros_like(visualization)
                
                # Convert to 3 channels and create color overlay
                roi_mask[y1:y2, x1:x2] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                
                # Create a color overlay only within the bounding box
                color_overlay = np.zeros_like(visualization)
                color_overlay[y1:y2, x1:x2] = color
                
                # Create a mask where cracks are detected (only within bbox)
                crack_mask = cv2.bitwise_and(roi_mask, color_overlay)
                
                # Create a mask for the bounding box area
                bbox_mask = np.zeros_like(visualization)
                bbox_mask[y1:y2, x1:x2] = 255
                
                # Apply the overlay only within the bounding box
                alpha = 0.4  # Reduced opacity for better visibility
                beta = 1 - alpha
                # First blend the crack mask with the original image
                blended = cv2.addWeighted(visualization, beta, crack_mask, alpha, 0)
                # Then use the bbox mask to combine with original image
                mask_inv = cv2.bitwise_not(bbox_mask)
                visualization = cv2.bitwise_and(visualization, mask_inv)
                visualization = cv2.add(visualization, cv2.bitwise_and(blended, bbox_mask))
                
                # Draw bounding box with thicker lines
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 3)
                
                # Create label with class name
                label = f"{class_name}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    visualization,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text in white
                cv2.putText(
                    visualization,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Calculate defect area for this detection
                defect_pixels, roi_pixels, relative_area = self.calculate_defect_area(
                    undistorted_img[y1:y2, x1:x2],
                    (0, 0, x2 - x1, y2 - y1),
                    (w, h)
                )
                total_defect_pixels += defect_pixels
                total_bbox_pixels += roi_pixels
            
            # Calculate severity metrics with all required inputs for fuzzy logic
            severity_level, real_world_area, total_bbox_pixels, avg_length_cm, avg_width_cm, defect_ratio = self.calculate_severity(
                undistorted_img, detections, distance_to_object_m
            )
            
            # Initialize defect counts
            defect_counts = {
                'Linear-Crack': 0,
                'Alligator-Crack': 0,
                'pothole': 0,
                'Total': len(detections)
            }
            
            # Count defects by class type
            for detection in detections:
                class_name = detection['class']
                if class_name in defect_counts:
                    defect_counts[class_name] += 1
            
            # Determine dominant defect type for fuzzy logic
            defect_type_counts = {class_name: 0 for class_name in ['Linear-Crack', 'Alligator-Crack', 'pothole']}
            for detection in detections:
                class_name = detection['class']
                if class_name in defect_type_counts:
                    defect_type_counts[class_name] += 1
            
            dominant_defect_type = max(defect_type_counts.items(), key=lambda x: x[1])[0]
            defect_type = 'crack' if 'Crack' in dominant_defect_type else 'pothole'
            
            # Calculate fuzzy severity using the fuzzy_logic module's main interface
            try:
                # Clamp values to valid ranges
                avg_length_cm = max(0, min(100, avg_length_cm))
                avg_width_cm = max(0, min(100, avg_width_cm))
                defect_ratio = max(0, min(1.0, defect_ratio))

                logger.info(f"Fuzzy logic inputs - Length: {avg_length_cm:.2f}cm, "
                            f"Width: {avg_width_cm:.2f}cm, "
                            f"Defect Ratio: {defect_ratio:.2f}, "
                            f"Type: {defect_type}")

                # Use the correct function name and parameters
                fuzzy_severity = calculate_severity_percentage(
                    length_cm=avg_length_cm,
                    width_cm=avg_width_cm,
                    defect_ratio=defect_ratio,
                    defect_type=dominant_defect_type  # Use the actual defect type name
                )
                
                # Calculate repair decision using random forest
                try:
                    # Get traffic volume (use default if not available)
                    traffic_volume = 5000.0  # Default medium traffic
                    
                    # Use random forest model with fuzzy severity as input
                    repair_decision = predict_repair_probability(
                        road_volume=traffic_volume,
                        defect_ratio=defect_ratio,
                        severity_level=fuzzy_severity/100.0,  # Convert to 0-1 range
                        threshold=0.25  # Lower threshold to be more sensitive to defects
                    )
                    
                    if MODEL_AVAILABLE:
                        logger.info(f"Using random forest model for repair decision: {repair_decision}")
                    else:
                        logger.info(f"Using fallback model for repair decision: {repair_decision}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get repair decision: {e}")
                    # Fallback to simple heuristic using fuzzy severity
                    repair_decision = 1 if (fuzzy_severity > 20.0 or defect_ratio > 0.1) else 0
                    logger.info(f"Using emergency fallback decision based on fuzzy severity {fuzzy_severity:.1f}% and defect ratio {defect_ratio:.3f}")

                logger.info(f"Fuzzy severity calculation successful: {fuzzy_severity:.2f}%")
            except Exception as e:
                logger.warning(f"Fuzzy logic calculation failed: {e}")
                logger.exception("Full traceback:")
                fuzzy_severity = severity_level * 100
                repair_decision = 0
            
            # Initialize defect counts
            defect_counts = {
                'Linear-Crack': 0,
                'Alligator-Crack': 0,
                'pothole': 0,
                'Total': len(detections)
            }
            
            # Count defects by class type
            for detection in detections:
                class_name = detection['class']
                if class_name in defect_counts:
                    defect_counts[class_name] += 1
            
            # Add metadata panel
            panel_height = 60
            panel = np.full((panel_height, visualization.shape[1], 3), 200, dtype=np.uint8)
            
            # Create summary text
            summary_text = f"Total Defects: {len(detections)} | "
            for defect_type, count in defect_counts.items():
                if defect_type != 'Total':
                    summary_text += f"{defect_type}: {count} | "
            summary_text += f"Severity: {severity_level:.2f} | Area: {real_world_area:.5f} m²"
            
            if gps_loc:
                summary_text += f" | GPS: ({gps_loc[0]:.6f}, {gps_loc[1]:.6f})"
            
            # Draw text on panel
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 0, 0)
            thickness = 2
            
            # Split text into two lines if it's too long
            if len(summary_text) > 80:
                parts = summary_text.split(' | ')
                line1 = ' | '.join(parts[:len(parts)//2])
                line2 = ' | '.join(parts[len(parts)//2:])
                
                text_size1 = cv2.getTextSize(line1, font, font_scale, thickness)[0]
                text_x1 = (panel.shape[1] - text_size1[0]) // 2
                cv2.putText(panel, line1, (text_x1, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                text_size2 = cv2.getTextSize(line2, font, font_scale, thickness)[0]
                text_x2 = (panel.shape[1] - text_size2[0]) // 2
                cv2.putText(panel, line2, (text_x2, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
            else:
                text_size = cv2.getTextSize(summary_text, font, font_scale, thickness)[0]
                text_x = (panel.shape[1] - text_size[0]) // 2
                cv2.putText(panel, summary_text, (text_x, 35), font, font_scale, font_color, thickness, cv2.LINE_AA)
            
            # Combine visualization with panel
            final_img = np.vstack([visualization, panel])

            # Save processed image
            cv2.imwrite(save_path, final_img)
            logger.info(f"Saved processed image to: {save_path}")

            # Update metadata with all calculated values
            output_metadata = {
                'GPSLocation': gps_loc,
                'SeverityLevel': float(severity_level),
                'RealWorldArea': float(real_world_area),
                'DefectPixelCount': int(total_defect_pixels),
                'TotalPixelCount': int(total_bbox_pixels),
                'DistanceToObject': float(distance_to_object_m),
                'ImageShape': [int(x) for x in undistorted_img.shape],
                'ProcessingTimestamp': datetime.now().isoformat(),
                'FuzzySeverity': float(fuzzy_severity),
                'RepairDecision': int(repair_decision),
                'DefectCounts': {k: int(v) for k, v in defect_counts.items()},
                'AverageLength': float(avg_length_cm),
                'AverageWidth': float(avg_width_cm),
                'DefectRatio': float(defect_ratio),
                'DominantDefectType': str(dominant_defect_type),
            }
            
            # Save metadata as JSON
            json_path = os.path.splitext(save_path)[0] + '_metadata.json'
            with open(json_path, 'w') as f:
                json.dump(output_metadata, f, indent=4)
            logger.info(f"Saved metadata to: {json_path}")
            
            return final_img, [total_defect_pixels], defect_ratio, output_metadata

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.exception("Full traceback:")
            return None, [], 0.0, {}

    def process_images(self, image_paths: List[str], camera_matrix: np.ndarray, 
                       distortion_coeffs: np.ndarray, save_dir: str, 
                       distance_to_object_m: float = 1.0, 
                       confidence_threshold: float = 0.15, 
                       batch_size: int = 8,
                       auto_delete_raw_failed_detections = False) -> List[Tuple[Optional[np.ndarray], List[int], float, Dict]]:
        """
        Process multiple images in batches using multithreading.
        
        Args:
            image_paths: List of image file paths to process.
            camera_matrix: Camera calibration matrix.
            distortion_coeffs: Distortion coefficients.
            save_dir: Directory to save processed images.
            distance_to_object_m: Distance to object in meters.
            confidence_threshold: Confidence threshold for defect detection.
            batch_size: Number of images to process in parallel.
        
        Returns:
            List of results for each processed image.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_image = {executor.submit(self.process_image, image_path, camera_matrix, 
                                                distortion_coeffs, os.path.join(save_dir, os.path.basename(image_path)), 
                                                distance_to_object_m, confidence_threshold, auto_delete_raw_failed_detections): image_path 
                                for image_path in image_paths}
            
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    if auto_delete_raw_failed_detections:
                        try:
                            os.remove(image_path)
                            logger.info(f"Deleted raw image {image_path} due to processing failure")
                        except Exception as delete_error:
                            logger.error(f"Failed to delete image {image_path}: {delete_error}")
                    results.append((None, [], 0.0, {}))  # Append a failure result

        return results


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))



if __name__ == "__main__":
    # === Load Image ===
    image_path = "C:/Users/bentl/Desktop/RoadDefectSystem/Detection/detection_20250620_070232.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # === Define Camera Parameters ===
    camera_matrix = np.array([
        [933.23, 0, 960],
        [0, 711, 540],
        [0, 0, 1]
    ], dtype=np.float32)
    distortion_coeffs = np.zeros(5)  # Assuming no distortion

    # === Apply Wide-Angle Warp (Barrel Distortion) ===
    wide_angle_img = apply_barrel_distortion(img, k1=-0.3)

    # === Apply Pinhole Camera Model (Undistortion) ===
    undistorted = apply_pinhole_camera_model(img, camera_matrix, distortion_coeffs)

    # === Show Distorted and Undistorted ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(cv2.cvtColor(wide_angle_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Simulated Wide-Angle Image (Barrel Distortion)")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axs[1].set_title("After Pinhole Camera Model")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()