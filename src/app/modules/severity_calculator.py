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
from skimage.segmentation import slic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_image_metadata(image_path: str) -> Tuple[Dict, Optional[Tuple[float, float]]]:
    """
    Extract metadata and GPS information from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing (metadata_dict, gps_location)
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return {}, None
                
            metadata = {}
            gps_location = None
            
            # Extract basic EXIF data
            for tag_id in exif:
                tag = Image.TAGS.get(tag_id, tag_id)
                data = exif.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode('utf-8', errors='replace')
                metadata[tag] = data
                
            # Extract GPS data if available
            if 34853 in exif:  # GPSInfo tag
                gps_info = exif[34853]
                try:
                    lat_data = gps_info.get(2)
                    lon_data = gps_info.get(4)
                    lat_ref = gps_info.get(1)
                    lon_ref = gps_info.get(3)
                    
                    if all([lat_data, lon_data, lat_ref, lon_ref]):
                        lat = float(lat_data[0]) + float(lat_data[1])/60 + float(lat_data[2])/3600
                        lon = float(lon_data[0]) + float(lon_data[1])/60 + float(lon_data[2])/3600
                        
                        if lat_ref == 'S': lat = -lat
                        if lon_ref == 'W': lon = -lon
                        
                        gps_location = (lat, lon)
                except Exception as e:
                    logger.warning(f"Error parsing GPS data: {e}")
                    
            return metadata, gps_location
            
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {}, None

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

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_contrast(gray_image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using histogram equalization."""
    gray_eq = exposure.equalize_hist(gray_image)
    return (gray_eq * 255).astype(np.uint8)

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

class SeverityCalculator:
    """
    Calculate the severity of road defects based on image analysis.
    """
    
    def __init__(self, camera_width: int, camera_height: int, 
                 focal_length: float = 35.0,
                 sensor_width: float = 36.0,
                 sensor_height: float = 24.0):
        """
        Initialize the severity calculator.
        
        Args:
            camera_width: Camera sensor width in pixels
            camera_height: Camera sensor height in pixels
            focal_length: Camera focal length in mm
            sensor_width: Physical sensor width in mm
            sensor_height: Physical sensor height in mm
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

    def calculate_defect_area(self, 
                            image_roi_bgr: np.ndarray,
                            bbox_relative_to_roi: Tuple[int, int, int, int] = (0,0,-1,-1)
                            ) -> Tuple[int, int]:
        """
        Calculate defect area within the provided image ROI.
        
        Args:
            image_roi_bgr: Input image in BGR format
            bbox_relative_to_roi: Bounding box coordinates (x1,y1,x2,y2)
            
        Returns:
            Tuple of (defect_area_pixels, total_processed_area_pixels)
        """
        if image_roi_bgr is None or image_roi_bgr.size == 0:
            return 0, 0

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
            return 0, 0

        # Apply thresholding and noise reduction
        var_mask = self._apply_variance_thresholding(gray_roi_processed)
        nib_mask = self._apply_niblack_thresholding(gray_roi_processed)
        
        combined_mask = cv2.bitwise_or(var_mask, nib_mask)
        clean_mask = self._reduce_noise(combined_mask)
        
        defect_area_pixels = np.sum(clean_mask == 255)
        total_processed_area_pixels = gray_roi_processed.size
        
        return defect_area_pixels, total_processed_area_pixels

    def _apply_variance_thresholding(self, gray_roi: np.ndarray, 
                                   var_thresh_val: int = 10) -> np.ndarray:
        """Apply variance-based thresholding."""
        if gray_roi.size == 0:
            return np.array([], dtype=np.uint8)
            
        _, thresh_img = cv2.threshold(gray_roi, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variance = np.var(gray_roi)
        
        if variance < var_thresh_val:
            return np.zeros_like(gray_roi, dtype=np.uint8)
        return thresh_img

    def _apply_niblack_thresholding(self, gray_roi: np.ndarray,
                                  window_size: int = 35,
                                  k: float = 0.1) -> np.ndarray:
        """Apply Niblack thresholding."""
        if gray_roi.size == 0:
            return np.array([], dtype=np.uint8)
            
        min_dim = min(gray_roi.shape)
        if min_dim < window_size:
            window_size = max(3, min_dim if min_dim % 2 != 0 else min_dim - 1)

        thresh_values = threshold_niblack(gray_roi, window_size=window_size, k=k)
        binary_img = gray_roi <= thresh_values
        return (binary_img * 255).astype(np.uint8)

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
                         ) -> Tuple[float, float, int]:
        """
        Calculate severity metrics for detected defects.
        
        Args:
            image: Input image
            detections: List of detection dictionaries with 'bbox' key
            distance_to_object_m: Distance to object in meters
            
        Returns:
            Tuple of (severity_level, real_world_area_m2, total_bbox_area)
        """
        if not detections:
            return 0.0, 0.0, 0.0
            
        total_defect_pixels = 0
        total_bbox_pixels = 0
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0:
                continue
                
            # Ensure ROI coordinates are within bounds
            img_h, img_w = image.shape[:2]
            roi_x1, roi_y1 = max(0, x1), max(0, y1)
            roi_x2, roi_y2 = min(img_w, x2), min(img_h, y2)
            
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                continue
                
            current_roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            defect_pixels, roi_pixels = self.calculate_defect_area(current_roi)
            
            total_defect_pixels += defect_pixels
            total_bbox_pixels += roi_pixels
            
        # Calculate severity metrics
        avg_defect_ratio = (total_defect_pixels / total_bbox_pixels 
                          if total_bbox_pixels > 0 else 0.0)
        real_world_area = self.calculate_real_world_size(total_defect_pixels, 
                                                       distance_to_object_m)
        severity_level = min(1.0, avg_defect_ratio * 2.0)
        
        return severity_level, real_world_area, total_bbox_pixels

    def process_image(self,
                     image_path: str,
                     camera_matrix: np.ndarray,
                     distortion_coeffs: np.ndarray,
                     save_path: str,
                     distance_to_object_m: float = 1.0
                     ) -> Tuple[Optional[np.ndarray], List[int], float, Dict]:
        """
        Process an image to detect and analyze road defects, overlaying a mask and embedding metadata.
        """
        try:
            # Extract metadata
            metadata, gps_loc = extract_image_metadata(image_path)
            
            # Load and undistort image
            original_img = cv2.imread(image_path)
            if original_img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None, [], 0.0, {}
            undistorted_img = apply_pinhole_camera_model(original_img, camera_matrix, distortion_coeffs)
            
            # Process image
            gray_img = convert_to_grayscale(undistorted_img)
            contrast_img = enhance_contrast(gray_img)
            edges = apply_canny_edge_detection(contrast_img)
            
            # Defect mask: use Niblack thresholding for cracks/defects
            defect_mask = self._apply_niblack_thresholding(gray_img)
            # Optionally combine with edge mask for more precision
            # defect_mask = cv2.bitwise_and(defect_mask, edges)
            defect_mask = self._reduce_noise(defect_mask)

            # Calculate severity metrics (use the whole image as one region)
            defect_pixels = np.sum(defect_mask == 255)
            total_pixels = defect_mask.size
            avg_defect_ratio = defect_pixels / total_pixels if total_pixels > 0 else 0.0
            real_world_area = self.calculate_real_world_size(defect_pixels, distance_to_object_m)
            severity_level = min(1.0, avg_defect_ratio * 2.0)

            # Create colored overlay for mask (yellow)
            overlay = undistorted_img.copy()
            yellow = np.array([0, 255, 255], dtype=np.uint8)  # BGR for yellow
            mask_indices = defect_mask == 255
            overlay[mask_indices] = (0.6 * overlay[mask_indices] + 0.4 * yellow).astype(np.uint8)
            # Blend overlay with original for semi-transparency
            blended = cv2.addWeighted(undistorted_img, 0.7, overlay, 0.3, 0)

            # Add metadata panel at the bottom
            panel_height = 50
            panel = np.full((panel_height, blended.shape[1], 3), 200, dtype=np.uint8)
            # Compose metadata text
            text = f"Severity: {severity_level:.2f}   Defect Area: {real_world_area:.5f} m²"
            if gps_loc:
                text += f"   GPS: ({gps_loc[0]:.6f}, {gps_loc[1]:.6f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 0)
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (panel.shape[1] - text_size[0]) // 2
            text_y = (panel_height + text_size[1]) // 2
            cv2.putText(panel, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
            # Stack panel below image
            final_img = np.vstack([blended, panel])

            # Save processed image
            cv2.imwrite(save_path, final_img)

            # Prepare metadata
            output_metadata = {
                'GPSLocation': gps_loc,
                'SeverityLevel': severity_level,
                'RealWorldArea': real_world_area,
                'DefectPixelCount': int(defect_pixels),
                'TotalPixelCount': int(total_pixels),
                'DistanceToObject': distance_to_object_m,
                'ImageShape': undistorted_img.shape,
                'ProcessingTimestamp': datetime.now().isoformat()
            }
            # Save metadata as JSON (optional, for record-keeping)
            json_path = os.path.splitext(save_path)[0] + '_metadata.json'
            with open(json_path, 'w') as f:
                json.dump(output_metadata, f, indent=4)

            return final_img, [defect_pixels], avg_defect_ratio, output_metadata
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, [], 0.0, {}