import cv2
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, dilation
from skimage.filters import threshold_niblack
from skimage.segmentation import slic
from skimage.color import label2rgb

def apply_variance_thresholding(image, window_size=25, k=-0.2):
    """
    Apply variance-based thresholding to detect defects.
    
    Args:
        image: Input image
        window_size: Size of the window for local variance calculation
        k: Threshold multiplier
        
    Returns:
        Binary mask of detected defects
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = cv2.blur(gray, (window_size, window_size))
    sq_mean = cv2.blur(np.square(gray), (window_size, window_size))
    variance = np.sqrt(sq_mean - np.square(mean))
    local_threshold = mean + k * variance
    binary = gray < local_threshold
    binary_cleaned = remove_small_objects(binary, min_size=100)
    binary_cleaned = remove_small_holes(binary_cleaned, area_threshold=100)
    binary_cleaned = dilation(binary_cleaned, footprint=np.ones((3, 3)))
    return (binary_cleaned * 255).astype(np.uint8)

def apply_niblack_thresholding(image, window_size=15, k=-0.2, use_dilation=False):
    """
    Apply Niblack thresholding to detect defects.
    
    Args:
        image: Input image
        window_size: Size of the window for local threshold calculation
        k: Threshold multiplier
        use_dilation: Whether to apply dilation to the result
        
    Returns:
        Binary mask of detected defects
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)
    gray = cv2.convertScaleAbs(cv2.addWeighted(gray, 1.5, edges, -0.5, 0))
    thresh_niblack = threshold_niblack(gray, window_size=window_size, k=k)
    binary_niblack = gray < thresh_niblack
    binary_niblack = remove_small_objects(binary_niblack, min_size=100)
    binary_niblack = remove_small_holes(binary_niblack, area_threshold=100)
    if use_dilation:
        binary_niblack = dilation(binary_niblack, footprint=np.ones((3, 3)))
    return (binary_niblack * 255).astype(np.uint8)

def reduce_noise(mask, kernel_size=3, min_size=50):
    """
    Reduce noise in a binary mask using morphological operations.
    
    Args:
        mask: Input binary mask
        kernel_size: Size of the morphological operation kernel
        min_size: Minimum size of objects to keep
        
    Returns:
        Cleaned binary mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, min_size=min_size)
    mask_bool = remove_small_holes(mask_bool, area_threshold=min_size)
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    return mask_uint8

def overlay_cracks_only(original_image, binary_crack_mask, alpha=0.5):
    """
    Overlay detected cracks on the original image.
    
    Args:
        original_image: Original RGB image
        binary_crack_mask: Binary mask of detected cracks
        alpha: Transparency of the overlay
        
    Returns:
        Image with overlaid cracks
    """
    mask = (binary_crack_mask == 255).astype(np.uint8)
    overlay = np.zeros_like(original_image)
    overlay[:, :] = [0, 0, 255]  # Blue
    result = original_image.copy()
    result[mask == 1] = cv2.addWeighted(original_image[mask == 1], 1 - alpha, overlay[mask == 1], alpha, 0)
    return result

class ImagePreprocessor:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def preprocess(self, image):
        """Apply preprocessing steps to the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply contrast enhancement
        enhanced = self.enhance_contrast(equalized)
        
        # Apply Canny edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Apply Niblack thresholding
        binary = self.niblack_thresholding(enhanced)
        
        # Apply superpixel segmentation
        segmented = self.apply_superpixels(image)
        
        # Overlay segmentation on original image
        result = self.overlay_segmentation(image, segmented)
        
        return result
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def niblack_thresholding(self, image, window_size=15, k=0.2):
        """Apply Niblack thresholding"""
        mean = cv2.blur(image, (window_size, window_size))
        mean_sq = cv2.blur(image * image, (window_size, window_size))
        std = np.sqrt(mean_sq - mean * mean)
        threshold = mean + k * std
        return (image > threshold).astype(np.uint8) * 255
    
    def apply_superpixels(self, image, n_segments=100):
        """Apply SLIC superpixel segmentation"""
        # Convert to RGB for SLIC
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply SLIC
        segments = slic(rgb, n_segments=n_segments, compactness=10)
        
        # Convert segments to RGB for visualization
        return label2rgb(segments, rgb, kind='avg')
    
    def overlay_segmentation(self, original, segmented):
        """Overlay segmentation on original image"""
        # Convert segmented image to BGR
        segmented_bgr = cv2.cvtColor(segmented.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Blend images
        alpha = 0.7
        return cv2.addWeighted(original, 1-alpha, segmented_bgr, alpha, 0)
    
    def undistort_image(self, image):
        """Apply pinhole camera model to undistort the image"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
            
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def set_camera_parameters(self, camera_matrix, dist_coeffs):
        """Set camera calibration parameters"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs 