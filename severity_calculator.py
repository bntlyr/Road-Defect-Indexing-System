import cv2
import numpy as np
from preprocessing_severity import apply_variance_thresholding, apply_niblack_thresholding, reduce_noise

class SeverityCalculator:
    def __init__(self, camera_width, camera_height, focal_length=35, sensor_width=36, sensor_height=24):
        """
        Initialize the severity calculator with camera parameters.
        
        Args:
            camera_width: Width of the camera image in pixels
            camera_height: Height of the camera image in pixels
            focal_length: Focal length of the camera in mm (default: 35mm)
            sensor_width: Width of the camera sensor in mm (default: 36mm for full frame)
            sensor_height: Height of the camera sensor in mm (default: 24mm for full frame)
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        
        # Calculate pixel size in mm
        self.pixel_size_x = sensor_width / camera_width
        self.pixel_size_y = sensor_height / camera_height
        
        # Calculate field of view angles
        self.fov_x = 2 * np.arctan(sensor_width / (2 * focal_length))
        self.fov_y = 2 * np.arctan(sensor_height / (2 * focal_length))
    
    def calculate_defect_area(self, image, bbox):
        """
        Calculate the area of defects within a bounding box.
        
        Args:
            image: The input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            defect_area: Area of defects in pixels
            total_area: Total area of the bounding box in pixels
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # Apply both thresholding methods
        var_mask = apply_variance_thresholding(roi)
        nib_mask = apply_niblack_thresholding(roi)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(var_mask, nib_mask)
        
        # Reduce noise
        clean_mask = reduce_noise(combined_mask, kernel_size=5, min_size=100)
        
        # Calculate areas
        defect_area = np.sum(clean_mask == 255)
        total_area = (x2 - x1) * (y2 - y1)
        
        return defect_area, total_area
    
    def calculate_real_world_size(self, pixel_area, distance_to_object):
        """
        Calculate the real-world size of a defect using the pinhole camera model.
        
        Args:
            pixel_area: Area in pixels
            distance_to_object: Distance to the object in meters
            
        Returns:
            real_world_area: Area in square meters
        """
        # Convert pixel area to sensor area (mm²)
        sensor_area = pixel_area * self.pixel_size_x * self.pixel_size_y
        
        # Calculate real-world area using similar triangles
        # Area scales with the square of the distance
        real_world_area = (sensor_area * (distance_to_object * 1000)**2) / (self.focal_length**2)
        
        # Convert to square meters
        real_world_area = real_world_area / 1e6
        
        return real_world_area
    
    def calculate_severity(self, image, detections, distance_to_object=1.0):
        """
        Calculate the severity level of defects in an image.
        
        Args:
            image: The input image
            detections: List of detection dictionaries with 'bbox' and 'class_name'
            distance_to_object: Distance to the object in meters
            
        Returns:
            severity_level: Normalized severity level (0-1)
            real_world_area: Total real-world area of defects in square meters
        """
        if not detections:
            return 0.0, 0.0
        
        total_defect_area = 0
        total_bbox_area = 0
        
        # Calculate defect areas for each detection
        for detection in detections:
            defect_area, bbox_area = self.calculate_defect_area(image, detection['bbox'])
            total_defect_area += defect_area
            total_bbox_area += bbox_area
        
        # Calculate average defect area ratio
        if total_bbox_area > 0:
            avg_defect_ratio = total_defect_area / total_bbox_area
        else:
            avg_defect_ratio = 0
        
        # Calculate real-world area
        real_world_area = self.calculate_real_world_size(total_defect_area, distance_to_object)
        
        # Normalize severity level (0-1)
        severity_level = min(1.0, avg_defect_ratio * 2)  # Scale by 2 to make it more sensitive
        
        return severity_level, real_world_area 