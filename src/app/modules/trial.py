import os
import numpy as np
import cv2
import logging
from src.app.modules.cloud_connector import CloudStorage
from src.app.modules.severity_calculator import SeverityCalculator
from src.app.modules import traffic_getter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(image_path, save_path, camera_matrix, distortion_coeffs, cloud=None, calculator=None, bboxes=None):
    """
    Run the full processing pipeline: process image, save locally, and upload to cloud if connected.
    """
    if calculator is None:
        calculator = SeverityCalculator(camera_width=1280, camera_height=720)
    final_img, defect_pixels, avg_defect_ratio, metadata = calculator.process_image(
        image_path=image_path,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        save_path=save_path
    )
    if final_img is not None:
        # Draw bounding boxes from metadata if available
        bboxes_to_draw = metadata.get("BoundingBoxes", bboxes)
        if bboxes_to_draw:
            img_with_boxes = final_img.copy()
            for bbox in bboxes_to_draw:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red box
            final_img = img_with_boxes

        # Save the processed image locally
        cv2.imwrite(save_path, final_img)
        logger.info(f"Image processed. Severity: {metadata.get('SeverityLevel'):.2f}, Area: {metadata.get('RealWorldArea'):.5f} m²")
        # Upload to cloud if connected
        if cloud and cloud.is_initialized:
            image_bgr = cv2.imread(save_path)
            if image_bgr is not None:
                cloud.upload_detection(
                    image=image_bgr,
                    defect_counts={'defect_pixels': sum(defect_pixels)},
                    frame_counts={'total_pixels': metadata.get("TotalPixelCount", 0)},
                    bboxes=bboxes
                )
    else:
        logger.error("Image processing failed.")

def main():
    # === Settings ===
    image_path = "test_images/detection_20250507_110722.png"
    save_path = "output/processed_road.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Dummy calibration (identity matrix and zero distortion)
    camera_matrix = np.array([[1000, 0, 640],
                              [0, 1000, 360],
                              [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.zeros(5)

    # === Initialize modules ===
    cloud = CloudStorage()
    calculator = SeverityCalculator(camera_width=1280, camera_height=720)

    # === Dummy bounding boxes for demonstration ===
    # Replace with actual detection results if available
    bboxes = [
        [100, 100, 400, 300],  # Example bbox [x1, y1, x2, y2]
        [500, 200, 700, 350]
    ]

    # === Process image ===
    run_pipeline(image_path, save_path, camera_matrix, distortion_coeffs, cloud, calculator, bboxes)

if __name__ == "__main__":
    main()
