"""
Test script for evaluating the accuracy of the SeverityCalculator module.
This script tests the masking accuracy and detection performance against ground truth data.

Usage:
    python test_severity_calculator_accuracy.py
    
Expected directory structure:
    C:\\Users\\rafab\\OneDrive\\Desktop\\accuracy test\\
    ├── image1.jpg
    ├── image1.txt
    ├── image2.jpg
    ├── image2.txt
    ├── image3.jpg
    └── image3.txt

Ground truth format (YOLO format):
    class_id x_center y_center width height
    Where coordinates are normalized (0-1)
"""

import cv2
import numpy as np
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.spatial.distance import cdist
import argparse

# Import the SeverityCalculator
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from modules.severity_calculator import SeverityCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyEvaluator:
    """
    Evaluates the accuracy of the SeverityCalculator against ground truth data.
    """
    
    def __init__(self, test_directory: str, model_path: str = None):
        """
        Initialize the accuracy evaluator.
        
        Args:
            test_directory: Directory containing test images and ground truth files
            model_path: Path to the YOLO model for defect detection
        """
        self.test_directory = test_directory
        self.model_path = model_path
        
        # Initialize severity calculator with default camera parameters
        self.calculator = SeverityCalculator(
            camera_width=1920,
            camera_height=1080,
            focal_length=35.0,
            sensor_width=36.0,
            sensor_height=24.0,
            model_path=model_path
        )
        
        # Default camera parameters (can be adjusted)
        self.camera_matrix = np.array([
            [933.23, 0, 960],
            [0, 711, 540],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.distortion_coeffs = np.zeros(5)
        
        # Class mapping for YOLO format
        self.class_names = {
            0: 'Linear-Crack',
            1: 'Alligator-Crack', 
            2: 'pothole'
        }
        
        # Results storage
        self.results = {
            'detection_accuracy': [],
            'mask_accuracy': [],
            'iou_scores': [],
            'pixel_accuracy': [],
            'severity_accuracy': []
        }
    
    def load_ground_truth(self, txt_path: str, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Load ground truth annotations from YOLO format text file.
        
        Args:
            txt_path: Path to the annotation file
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of ground truth bounding boxes
        """
        ground_truth = []
        
        if not os.path.exists(txt_path):
            logger.warning(f"Ground truth file not found: {txt_path}")
            return ground_truth
        
        height, width = image_shape
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int((x_center - box_width/2) * width)
                        y1 = int((y_center - box_height/2) * height)
                        x2 = int((x_center + box_width/2) * width)
                        y2 = int((y_center + box_height/2) * height)
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, width-1))
                        y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width-1))
                        y2 = max(0, min(y2, height-1))
                        
                        ground_truth.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': self.class_names.get(class_id, 'unknown'),
                            'class_id': class_id
                        })
                        
        except Exception as e:
            logger.error(f"Error loading ground truth from {txt_path}: {e}")
        
        return ground_truth
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_detections_to_ground_truth(self, detections: List[Dict], 
                                       ground_truth: List[Dict],
                                       iou_threshold: float = 0.5) -> Tuple[List, List, List]:
        """
        Match detected bounding boxes to ground truth boxes using IoU.
        
        Args:
            detections: List of detected bounding boxes
            ground_truth: List of ground truth bounding boxes
            iou_threshold: Minimum IoU for a match
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        true_positives = []
        false_positives = []
        false_negatives = ground_truth.copy()
        
        for detection in detections:
            best_iou = 0
            best_match_idx = -1
            
            for i, gt in enumerate(false_negatives):
                iou = self.calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                matched_gt = false_negatives.pop(best_match_idx)
                true_positives.append({
                    'detection': detection,
                    'ground_truth': matched_gt,
                    'iou': best_iou
                })
            else:
                false_positives.append(detection)
        
        return true_positives, false_positives, false_negatives
    
    def evaluate_mask_accuracy(self, image: np.ndarray, 
                              detections: List[Dict],
                              ground_truth: List[Dict]) -> Dict:
        """
        Evaluate the accuracy of defect masking within detected regions.
        
        Args:
            image: Input image
            detections: List of detected defects
            ground_truth: List of ground truth defects
            
        Returns:
            Dictionary containing mask accuracy metrics
        """
        if not detections or not ground_truth:
            return {
                'pixel_accuracy': 0.0,
                'mean_iou': 0.0,
                'dice_coefficient': 0.0,
                'total_defect_pixels': 0,
                'total_gt_pixels': 0
            }
        
        # Create masks for detected and ground truth regions
        height, width = image.shape[:2]
        detected_mask = np.zeros((height, width), dtype=np.uint8)
        gt_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill ground truth mask
        for gt in ground_truth:
            x1, y1, x2, y2 = gt['bbox']
            gt_mask[y1:y2, x1:x2] = 255
        
        # Fill detected mask using the severity calculator's defect area calculation
        total_detected_pixels = 0
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Use the calculator's method to get defect area
                defect_pixels, roi_pixels, _ = self.calculator.calculate_defect_area(
                    roi, (0, 0, x2-x1, y2-y1), (width, height)
                )
                
                # Create a mask for this ROI based on defect detection
                roi_mask = np.zeros_like(roi[:, :, 0] if len(roi.shape) == 3 else roi)
                
                # Apply the calculator's internal methods to create mask
                if len(roi.shape) == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = roi
                
                # Use the calculator's thresholding methods
                var_mask = self.calculator._apply_variance_thresholding(gray_roi)
                nib_mask = self.calculator._apply_niblack_thresholding(gray_roi)
                combined_mask = cv2.bitwise_or(var_mask, nib_mask)
                clean_mask = self.calculator._reduce_noise(combined_mask)
                
                # Place the mask back into the full image
                if clean_mask.size > 0:
                    detected_mask[y1:y2, x1:x2] = clean_mask
                    total_detected_pixels += np.sum(clean_mask == 255)
        
        # Calculate metrics
        intersection = cv2.bitwise_and(detected_mask, gt_mask)
        union = cv2.bitwise_or(detected_mask, gt_mask)
        
        intersection_pixels = np.sum(intersection == 255)
        union_pixels = np.sum(union == 255)
        detected_pixels = np.sum(detected_mask == 255)
        gt_pixels = np.sum(gt_mask == 255)
        
        # Pixel accuracy
        total_pixels = height * width
        correct_pixels = np.sum((detected_mask == gt_mask))
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # IoU (Jaccard Index)
        iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0
        
        # Dice coefficient (F1 score for pixels)
        dice = (2 * intersection_pixels) / (detected_pixels + gt_pixels) if (detected_pixels + gt_pixels) > 0 else 0.0
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': iou,
            'dice_coefficient': dice,
            'total_defect_pixels': detected_pixels,
            'total_gt_pixels': gt_pixels,
            'intersection_pixels': intersection_pixels,
            'union_pixels': union_pixels
        }
    
    def test_single_image(self, image_path: str, confidence_threshold: float = 0.15) -> Dict:
        """
        Test the severity calculator on a single image.
        
        Args:
            image_path: Path to the test image
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            Dictionary containing test results
        """
        logger.info(f"Testing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {}
        
        # Load ground truth
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        ground_truth = self.load_ground_truth(txt_path, image.shape[:2])
        
        if not ground_truth:
            logger.warning(f"No ground truth found for {image_path}")
            return {}
        
        # Run detection
        detections = self.calculator.detect_defects(image, confidence_threshold)
        
        # Match detections to ground truth
        true_positives, false_positives, false_negatives = self.match_detections_to_ground_truth(
            detections, ground_truth
        )
        
        # Calculate detection metrics
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0.0
        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate mask accuracy
        mask_metrics = self.evaluate_mask_accuracy(image, detections, ground_truth)
        
        # Calculate severity metrics
        severity_level, real_world_area, _, avg_length_cm, avg_width_cm, defect_ratio = self.calculator.calculate_severity(
            image, detections
        )
        
        # Calculate average IoU for matched detections
        avg_iou = np.mean([tp['iou'] for tp in true_positives]) if true_positives else 0.0
        
        results = {
            'image_path': image_path,
            'detection_metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(len(true_positives)),
                'false_positives': int(len(false_positives)),
                'false_negatives': int(len(false_negatives)),
                'avg_iou': float(avg_iou)
            },
            'mask_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v 
                           for k, v in mask_metrics.items()},
            'severity_metrics': {
                'severity_level': float(severity_level),
                'real_world_area': float(real_world_area),
                'avg_length_cm': float(avg_length_cm),
                'avg_width_cm': float(avg_width_cm),
                'defect_ratio': float(defect_ratio)
            },
            'ground_truth_count': int(len(ground_truth)),
            'detection_count': int(len(detections))
        }
        
        logger.info(f"Results for {os.path.basename(image_path)}:")
        logger.info(f"  Detection - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        logger.info(f"  Masking - Pixel Accuracy: {mask_metrics['pixel_accuracy']:.3f}, IoU: {mask_metrics['mean_iou']:.3f}")
        logger.info(f"  Severity Level: {severity_level:.3f}")
        
        return results
    
    def run_evaluation_with_results_only(self, confidence_threshold: float = 0.15) -> Dict:
        """
        Run evaluation and return only the results without creating any visualizations.
        
        Args:
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            Dictionary containing overall evaluation results
        """
        logger.info(f"Starting evaluation on directory: {self.test_directory}")
        
        if not os.path.exists(self.test_directory):
            logger.error(f"Test directory not found: {self.test_directory}")
            return {}
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(self.test_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.test_directory, file)
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    image_files.append(image_path)
        
        if not image_files:
            logger.error("No valid image-annotation pairs found in test directory")
            return {}
        
        logger.info(f"Found {len(image_files)} test images with annotations")
        
        # Test each image
        all_results = []
        for image_path in image_files:
            result = self.test_single_image(image_path, confidence_threshold)
            if result:
                all_results.append(result)
        
        if not all_results:
            logger.error("No valid results obtained")
            return {}
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(all_results)
        
        # Save detailed results
        self.save_results(all_results, overall_metrics)
        
        return overall_metrics

    def create_image_comparisons_batch(self, confidence_threshold: float = 0.15) -> None:
        """
        Create image comparisons for all test images as a separate process.
        
        Args:
            confidence_threshold: Confidence threshold for detection (not used in visualization)
        """
        logger.info("Creating image comparisons for all test images...")
        
        if not os.path.exists(self.test_directory):
            logger.error(f"Test directory not found: {self.test_directory}")
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(self.test_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.test_directory, file)
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    image_files.append(image_path)
        
        if not image_files:
            logger.error("No valid image-annotation pairs found in test directory")
            return
        
        logger.info(f"Creating visualizations for {len(image_files)} test images")
        
        # Create visualization for each image
        for image_path in image_files:
            result = self.test_single_image(image_path, confidence_threshold)
            if result:
                print(f"\nCreating visualization for: {os.path.basename(image_path)}")
                self.create_enhanced_image_comparison(image_path, result, save_visualization=True)

    def create_enhanced_image_comparison(self, image_path: str, result: Dict, save_visualization: bool = True) -> None:
        """
        Create an enhanced visual comparison showing original image, ground truth, detections with both 
        bounding boxes and masks, and accuracy metrics. Confidence threshold is not shown in visualization.
        
        Args:
            image_path: Path to the test image
            result: Test result dictionary
            save_visualization: Whether to save the visualization
        """
        try:
            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image for visualization: {image_path}")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Load ground truth
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            ground_truth = self.load_ground_truth(txt_path, image.shape[:2])
            
            # Run detection to get visual results (using low threshold for comprehensive detection)
            detections = self.calculator.detect_defects(image, 0.1)  # Low threshold for visualization
            
            # Get mask data from severity calculator
            try:
                # The calculate_severity method returns a tuple, not a dict with mask
                # We need to create a mask ourselves based on the detections
                defect_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Create mask from detected regions using the calculator's methods
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Apply the calculator's internal methods to create mask
                        if len(roi.shape) == 3:
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_roi = roi
                        
                        # Use the calculator's thresholding methods if available
                        try:
                            var_mask = self.calculator._apply_variance_thresholding(gray_roi)
                            nib_mask = self.calculator._apply_niblack_thresholding(gray_roi)
                            combined_mask = cv2.bitwise_or(var_mask, nib_mask)
                            clean_mask = self.calculator._reduce_noise(combined_mask)
                            
                            # Place the mask back into the full image
                            if clean_mask.size > 0:
                                defect_mask[y1:y2, x1:x2] = clean_mask
                        except:
                            # Fallback: just mark the bounding box area
                            defect_mask[y1:y2, x1:x2] = 255
                            
            except Exception as e:
                logger.warning(f"Could not generate mask: {e}")
                defect_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create the figure with subplots in 2x3 layout
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 16))
            
            # Colors for different defect types
            colors = {
                'Linear-Crack': 'red',
                'Alligator-Crack': 'blue', 
                'pothole': 'green',
                'unknown': 'orange'
            }
            
            # 1. Original Image
            ax1.imshow(image_rgb)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 2. Ground Truth Annotations
            ax2.imshow(image_rgb)
            for i, gt in enumerate(ground_truth):
                x1, y1, x2, y2 = gt['bbox']
                class_name = gt['class']
                color = colors.get(class_name, colors['unknown'])
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                
                # Add label without confidence
                ax2.text(x1, y1-5, f'GT: {class_name}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                        fontsize=10, color='white', weight='bold')
            
            ax2.set_title(f'Ground Truth ({len(ground_truth)} annotations)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # 3. Detected Bounding Boxes Only
            ax3.imshow(image_rgb)
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class']
                color = colors.get(class_name, colors['unknown'])
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color, facecolor='none', linestyle='--')
                ax3.add_patch(rect)
                
                # Add label without confidence (removed as requested)
                ax3.text(x1, y1-5, f'{class_name}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                        fontsize=10, color='white', weight='bold')
            
            ax3.set_title(f'Detected Bounding Boxes ({len(detections)} detected)', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            # 4. Detection Mask Overlay
            ax4.imshow(image_rgb)
            if defect_mask is not None and np.any(defect_mask > 0):
                # Create colored mask overlay
                mask_overlay = np.zeros((*defect_mask.shape, 3), dtype=np.uint8)
                mask_overlay[defect_mask > 0] = [255, 0, 0]  # Red for defects
                ax4.imshow(mask_overlay, alpha=0.4)
            
            ax4.set_title('Defect Mask Overlay', fontsize=14, fontweight='bold')
            ax4.axis('off')
            
            # 5. Combined: Bounding Boxes + Mask
            ax5.imshow(image_rgb)
            
            # Add mask overlay
            if defect_mask is not None and np.any(defect_mask > 0):
                mask_overlay = np.zeros((*defect_mask.shape, 3), dtype=np.uint8)
                mask_overlay[defect_mask > 0] = [255, 255, 0]  # Yellow for defects
                ax5.imshow(mask_overlay, alpha=0.3)
            
            # Add bounding boxes
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class']
                color = colors.get(class_name, colors['unknown'])
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color, facecolor='none', linestyle='-')
                ax5.add_patch(rect)
                
                # Add label
                ax5.text(x1, y1-5, f'{class_name}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                        fontsize=10, color='white', weight='bold')
            
            ax5.set_title('Combined: Bounding Boxes + Mask', fontsize=14, fontweight='bold')
            ax5.axis('off')
            
            # 6. Accuracy Metrics Display
            ax6.axis('off')
            
            # Extract metrics from result
            det_metrics = result['detection_metrics']
            mask_metrics = result['mask_metrics']
            
            # Create metrics text
            metrics_text = f"""
ACCURACY RESULTS

Detection Performance:
• Precision: {det_metrics['precision']:.3f} ({det_metrics['precision']*100:.1f}%)
• Recall: {det_metrics['recall']:.3f} ({det_metrics['recall']*100:.1f}%)
• F1-Score: {det_metrics['f1_score']:.3f} ({det_metrics['f1_score']*100:.1f}%)
• Avg IoU: {det_metrics['avg_iou']:.3f}

Detection Counts:
• True Positives: {det_metrics['true_positives']}
• False Positives: {det_metrics['false_positives']}
• False Negatives: {det_metrics['false_negatives']}

Masking Performance:
• Pixel Accuracy: {mask_metrics['pixel_accuracy']:.3f} ({mask_metrics['pixel_accuracy']*100:.1f}%)
• Mean IoU: {mask_metrics['mean_iou']:.3f} ({mask_metrics['mean_iou']*100:.1f}%)
• Dice Coefficient: {mask_metrics['dice_coefficient']:.3f}

Defect Analysis:
• Defect Pixels: {mask_metrics['total_defect_pixels']:,}
• Ground Truth Pixels: {mask_metrics['total_gt_pixels']:,}
"""
            
            # Add color-coded performance indicators
            overall_score = (det_metrics['f1_score'] + mask_metrics['pixel_accuracy']) / 2
            if overall_score >= 0.8:
                performance_color = 'green'
                performance_text = "EXCELLENT"
            elif overall_score >= 0.6:
                performance_color = 'orange'
                performance_text = "GOOD"
            elif overall_score >= 0.4:
                performance_color = 'red'
                performance_text = "FAIR"
            else:
                performance_color = 'darkred'
                performance_text = "POOR"
            
            # Display metrics
            ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            # Add overall performance indicator
            ax6.text(0.05, 0.05, f"OVERALL PERFORMANCE: {performance_text}\nScore: {overall_score:.3f}", 
                    transform=ax6.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment='bottom', color=performance_color,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=performance_color, alpha=0.2))
            
            # Add legend for defect types
            legend_elements = [patches.Patch(facecolor=color, label=defect_type) 
                             for defect_type, color in colors.items() if defect_type != 'unknown']
            ax6.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
            
            # Set overall title
            image_name = os.path.basename(image_path)
            fig.suptitle(f'Enhanced Accuracy Analysis: {image_name}', fontsize=18, fontweight='bold')
            
            plt.tight_layout()
            
            if save_visualization:
                # Save the visualization
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_base = os.path.splitext(os.path.basename(image_path))[0]
                viz_filename = f"enhanced_accuracy_comparison_{image_base}_{timestamp}.png"
                plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved enhanced accuracy comparison: {viz_filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating enhanced image comparison for {image_path}: {e}")
            logger.exception("Full traceback:")

    def calculate_overall_metrics(self, all_results: List[Dict]) -> Dict:
        """
        Calculate overall metrics from individual test results.
        
        Args:
            all_results: List of individual test results
            
        Returns:
            Dictionary containing overall metrics
        """
        if not all_results:
            return {}
        
        # Aggregate detection metrics
        total_tp = sum(r['detection_metrics']['true_positives'] for r in all_results)
        total_fp = sum(r['detection_metrics']['false_positives'] for r in all_results)
        total_fn = sum(r['detection_metrics']['false_negatives'] for r in all_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Aggregate mask metrics
        avg_pixel_accuracy = np.mean([r['mask_metrics']['pixel_accuracy'] for r in all_results])
        avg_mask_iou = np.mean([r['mask_metrics']['mean_iou'] for r in all_results])
        avg_dice = np.mean([r['mask_metrics']['dice_coefficient'] for r in all_results])
        
        # Aggregate IoU metrics
        avg_detection_iou = np.mean([r['detection_metrics']['avg_iou'] for r in all_results])
        
        # Calculate overall accuracy percentage
        mask_accuracy_percentage = avg_pixel_accuracy * 100
        detection_accuracy_percentage = overall_f1 * 100
        
        overall_metrics = {
            'test_summary': {
                'total_images': int(len(all_results)),
                'total_detections': int(sum(r['detection_count'] for r in all_results)),
                'total_ground_truth': int(sum(r['ground_truth_count'] for r in all_results))
            },
            'detection_performance': {
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1_score': float(overall_f1),
                'accuracy_percentage': float(detection_accuracy_percentage),
                'avg_iou': float(avg_detection_iou)
            },
            'masking_performance': {
                'pixel_accuracy': float(avg_pixel_accuracy),
                'accuracy_percentage': float(mask_accuracy_percentage),
                'mean_iou': float(avg_mask_iou),
                'dice_coefficient': float(avg_dice)
            },
            'individual_results': all_results
        }
        
        return overall_metrics
    
    def save_results(self, all_results: List[Dict], overall_metrics: Dict):
        """
        Save evaluation results to files.
        
        Args:
            all_results: List of individual test results
            overall_metrics: Overall evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert the overall metrics
        json_serializable_metrics = convert_numpy_types(overall_metrics)
        
        # Save JSON results
        results_file = f"severity_calculator_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(json_serializable_metrics, f, indent=4)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save summary report
        report_file = f"severity_calculator_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("SEVERITY CALCULATOR ACCURACY EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Directory: {self.test_directory}\n")
            f.write(f"Model Path: {self.model_path}\n\n")
            
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images Tested: {overall_metrics['test_summary']['total_images']}\n")
            f.write(f"Total Detections: {overall_metrics['test_summary']['total_detections']}\n")
            f.write(f"Total Ground Truth: {overall_metrics['test_summary']['total_ground_truth']}\n\n")
            
            f.write("DETECTION ACCURACY\n")
            f.write("-" * 20 + "\n")
            det_perf = overall_metrics['detection_performance']
            f.write(f"Precision: {det_perf['precision']:.3f} ({det_perf['precision']*100:.1f}%)\n")
            f.write(f"Recall: {det_perf['recall']:.3f} ({det_perf['recall']*100:.1f}%)\n")
            f.write(f"F1-Score: {det_perf['f1_score']:.3f} ({det_perf['f1_score']*100:.1f}%)\n")
            f.write(f"Average IoU: {det_perf['avg_iou']:.3f}\n")
            f.write(f"Overall Detection Accuracy: {det_perf['accuracy_percentage']:.1f}%\n\n")
            
            f.write("MASKING ACCURACY\n")
            f.write("-" * 20 + "\n")
            mask_perf = overall_metrics['masking_performance']
            f.write(f"Pixel Accuracy: {mask_perf['pixel_accuracy']:.3f} ({mask_perf['accuracy_percentage']:.1f}%)\n")
            f.write(f"Mean IoU: {mask_perf['mean_iou']:.3f} ({mask_perf['mean_iou']*100:.1f}%)\n")
            f.write(f"Dice Coefficient: {mask_perf['dice_coefficient']:.3f} ({mask_perf['dice_coefficient']*100:.1f}%)\n\n")
            
            f.write("INDIVIDUAL IMAGE RESULTS\n")
            f.write("-" * 25 + "\n")
            for result in all_results:
                img_name = os.path.basename(result['image_path'])
                det_metrics = result['detection_metrics']
                mask_metrics = result['mask_metrics']
                f.write(f"{img_name}:\n")
                f.write(f"  Detection F1: {det_metrics['f1_score']:.3f}\n")
                f.write(f"  Mask Accuracy: {mask_metrics['pixel_accuracy']:.3f}\n")
                f.write(f"  Mask IoU: {mask_metrics['mean_iou']:.3f}\n\n")
        
        logger.info(f"Summary report saved to: {report_file}")
    
    def create_visualization(self, overall_metrics: Dict):
        """
        Create visualization plots for the evaluation results.
        
        Args:
            overall_metrics: Overall evaluation metrics
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Detection performance
            det_metrics = ['Precision', 'Recall', 'F1-Score']
            det_values = [
                overall_metrics['detection_performance']['precision'],
                overall_metrics['detection_performance']['recall'],
                overall_metrics['detection_performance']['f1_score']
            ]
            
            ax1.bar(det_metrics, det_values, color=['blue', 'green', 'orange'])
            ax1.set_title('Detection Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(det_values):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # Masking performance
            mask_metrics = ['Pixel Accuracy', 'Mean IoU', 'Dice Coefficient']
            mask_values = [
                overall_metrics['masking_performance']['pixel_accuracy'],
                overall_metrics['masking_performance']['mean_iou'],
                overall_metrics['masking_performance']['dice_coefficient']
            ]
            
            ax2.bar(mask_metrics, mask_values, color=['red', 'purple', 'brown'])
            ax2.set_title('Masking Performance Metrics')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
            for i, v in enumerate(mask_values):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # Overall accuracy comparison
            categories = ['Detection\nAccuracy', 'Masking\nAccuracy']
            accuracies = [
                overall_metrics['detection_performance']['accuracy_percentage'],
                overall_metrics['masking_performance']['accuracy_percentage']
            ]
            
            bars = ax3.bar(categories, accuracies, color=['skyblue', 'lightcoral'])
            ax3.set_title('Overall Accuracy Comparison')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_ylim(0, 100)
            for bar, acc in zip(bars, accuracies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center')
            
            # Individual image performance
            individual_results = overall_metrics['individual_results']
            image_names = [os.path.basename(r['image_path']) for r in individual_results]
            mask_accuracies = [r['mask_metrics']['pixel_accuracy'] * 100 for r in individual_results]
            
            ax4.bar(range(len(image_names)), mask_accuracies, color='green', alpha=0.7)
            ax4.set_title('Per-Image Masking Accuracy')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_xlabel('Images')
            ax4.set_xticks(range(len(image_names)))
            ax4.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in image_names], rotation=45)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"severity_calculator_visualization_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualization saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def create_image_comparison(self, image_path: str, result: Dict, save_visualization: bool = True) -> None:
        """
        Legacy image comparison method - kept for backward compatibility.
        Use create_enhanced_image_comparison for new features.
        
        Args:
            image_path: Path to the test image
            result: Test result dictionary
            save_visualization: Whether to save the visualization
        """
        logger.warning("Using legacy visualization method. Consider using create_enhanced_image_comparison for better features.")
        self.create_enhanced_image_comparison(image_path, result, save_visualization)
    
    def run_full_evaluation(self, confidence_threshold: float = 0.15, show_comparisons: bool = True) -> Dict:
        """
        Run evaluation on all test images in the directory.
        Now separated: first get results, then optionally create visualizations.
        
        Args:
            confidence_threshold: Confidence threshold for detection
            show_comparisons: Whether to show individual image comparisons
            
        Returns:
            Dictionary containing overall evaluation results
        """
        # First, get the results without any visualizations
        logger.info("Step 1: Running accuracy evaluation (results only)...")
        overall_metrics = self.run_evaluation_with_results_only(confidence_threshold)
        
        if not overall_metrics:
            logger.error("Failed to obtain evaluation results")
            return {}
        
        # Display results summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        
        det_acc = overall_metrics['detection_performance']['accuracy_percentage']
        mask_acc = overall_metrics['masking_performance']['accuracy_percentage']
        f1_score = overall_metrics['detection_performance']['f1_score']
        mean_iou = overall_metrics['masking_performance']['mean_iou']
        
        print(f"Detection Accuracy: {det_acc:.1f}%")
        print(f"Masking Accuracy: {mask_acc:.1f}%")
        print(f"F1-Score: {f1_score:.3f}")
        print(f"Mean IoU: {mean_iou:.3f}")
        print(f"Total Images Tested: {overall_metrics['test_summary']['total_images']}")
        print("="*60)
        
        # Then, optionally create enhanced visualizations
        if show_comparisons:
            print("\nStep 2: Creating enhanced image comparisons...")
            print("Note: Confidence thresholds are not shown in visualizations")
            self.create_image_comparisons_batch(confidence_threshold)
        
        return overall_metrics
def main():
    """
    Main function to run the accuracy evaluation.
    """
    parser = argparse.ArgumentParser(description='Test Severity Calculator Accuracy')
    parser.add_argument('--test_dir', type=str, 
                       default=r'C:\Users\rafab\OneDrive\Desktop\accuracy test',
                       help='Directory containing test images and annotations')
    parser.add_argument('--model_path', type=str,
                       default=r'src\models\road_defect.pt',
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.30,
                       help='Confidence threshold for detection')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--show_comparisons', action='store_true', default=True,
                       help='Show individual image comparisons (default: True)')
    parser.add_argument('--no_comparisons', action='store_true',
                       help='Disable individual image comparisons')
    
    args = parser.parse_args()
    
    # Handle comparison display option
    show_comparisons = args.show_comparisons and not args.no_comparisons
    
    # Initialize evaluator
    evaluator = AccuracyEvaluator(args.test_dir, args.model_path)
    
    # Run evaluation
    logger.info("Starting Severity Calculator Accuracy Evaluation")
    overall_metrics = evaluator.run_full_evaluation(args.confidence, show_comparisons)
    
    if overall_metrics:
        # Print summary to console
        print("\n" + "="*60)
        print("SEVERITY CALCULATOR ACCURACY EVALUATION RESULTS")
        print("="*60)
        print(f"Detection Accuracy: {overall_metrics['detection_performance']['accuracy_percentage']:.1f}%")
        print(f"Masking Accuracy: {overall_metrics['masking_performance']['accuracy_percentage']:.1f}%")
        print(f"Overall F1-Score: {overall_metrics['detection_performance']['f1_score']:.3f}")
        print(f"Mean IoU (Masking): {overall_metrics['masking_performance']['mean_iou']:.3f}")
        print("="*60)
        
        # Create visualization if requested
        if args.visualize:
            evaluator.create_visualization(overall_metrics)
    else:
        logger.error("Evaluation failed - no results obtained")


if __name__ == "__main__":
    main()
