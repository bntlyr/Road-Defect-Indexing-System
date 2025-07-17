"""
Utility script for preparing test data for severity calculator accuracy evaluation.
This script helps convert annotations and set up the test environment.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict

def create_sample_annotation(image_path: str, defect_boxes: List[Tuple]) -> str:
    """
    Create a sample annotation file in YOLO format.
    
    Args:
        image_path: Path to the image file
        defect_boxes: List of tuples (class_id, x1, y1, x2, y2) in pixel coordinates
        
    Returns:
        Path to created annotation file
    """
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Create annotation file
    annotation_path = os.path.splitext(image_path)[0] + '.txt'
    
    with open(annotation_path, 'w') as f:
        for class_id, x1, y1, x2, y2 in defect_boxes:
            # Convert to YOLO format (normalized coordinates)
            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
    
    print(f"Created annotation file: {annotation_path}")
    return annotation_path

def visualize_annotations(image_path: str, annotation_path: str = None):
    """
    Visualize annotations on the image.
    
    Args:
        image_path: Path to the image file
        annotation_path: Path to annotation file (optional, auto-detected if None)
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Load annotations
    if annotation_path is None:
        annotation_path = os.path.splitext(image_path)[0] + '.txt'
    
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found: {annotation_path}")
        return
    
    # Class names
    class_names = {0: 'Linear-Crack', 1: 'Alligator-Crack', 2: 'pothole'}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    # Read and draw annotations
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height
                
                x1 = int(x_center - box_width/2)
                y1 = int(y_center - box_height/2)
                x2 = int(x_center + box_width/2)
                y2 = int(y_center + box_height/2)
                
                color = colors[class_id % len(colors)]
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                
                label = class_names.get(class_id, f'Class {class_id}')
                cv2.putText(image_rgb, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(f"Annotations for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

def setup_test_directory(base_dir: str = r'C:\Users\rafab\OneDrive\Desktop\accuracy test'):
    """
    Set up the test directory structure and provide instructions.
    
    Args:
        base_dir: Base directory for test setup
    """
    print("SEVERITY CALCULATOR TEST SETUP")
    print("=" * 40)
    print()
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir)
            print(f"✓ Created test directory: {base_dir}")
        except Exception as e:
            print(f"❌ Failed to create directory: {e}")
            return
    else:
        print(f"✓ Test directory exists: {base_dir}")
    
    # Check for existing files
    image_files = []
    annotation_files = []
    
    for file in os.listdir(base_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
        elif file.endswith('.txt'):
            annotation_files.append(file)
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(annotation_files)} annotation files")
    print()
    
    # Instructions
    print("TEST DATA REQUIREMENTS:")
    print("-" * 25)
    print("1. Place 3 test images in the directory:")
    print("   - image1.jpg (or .png, .jpeg, etc.)")
    print("   - image2.jpg")
    print("   - image3.jpg")
    print()
    print("2. Create corresponding annotation files:")
    print("   - image1.txt")
    print("   - image2.txt") 
    print("   - image3.txt")
    print()
    print("ANNOTATION FORMAT (YOLO format):")
    print("-" * 30)
    print("Each line: class_id x_center y_center width height")
    print("Where:")
    print("  - class_id: 0=Linear-Crack, 1=Alligator-Crack, 2=pothole")
    print("  - All coordinates are normalized (0.0 to 1.0)")
    print("  - x_center, y_center: center of bounding box")
    print("  - width, height: dimensions of bounding box")
    print()
    print("EXAMPLE annotation file content:")
    print("0 0.5 0.3 0.2 0.1")
    print("1 0.7 0.6 0.15 0.08")
    print()
    print("This means:")
    print("- Linear crack at center (50%, 30%) with size 20% x 10%")
    print("- Alligator crack at center (70%, 60%) with size 15% x 8%")
    print()
    
    # Create sample annotation file
    sample_annotation = os.path.join(base_dir, "sample_annotation.txt")
    with open(sample_annotation, 'w') as f:
        f.write("# Sample annotation file\n")
        f.write("# Format: class_id x_center y_center width height\n")
        f.write("# Class IDs: 0=Linear-Crack, 1=Alligator-Crack, 2=pothole\n")
        f.write("# Coordinates are normalized (0.0 to 1.0)\n")
        f.write("0 0.5 0.3 0.2 0.1\n")
        f.write("1 0.7 0.6 0.15 0.08\n")
    
    print(f"✓ Created sample annotation file: {sample_annotation}")
    print()
    print("TO RUN THE TEST:")
    print("-" * 15)
    print("1. Prepare your test images and annotations")
    print("2. Run: python test_severity_calculator_accuracy.py")
    print("3. Or run: python test_rdis_accuracy.py for a quick test")
    print()

def validate_test_data(test_dir: str = r'C:\Users\rafab\OneDrive\Desktop\accuracy test'):
    """
    Validate test data format and completeness.
    
    Args:
        test_dir: Directory containing test data
    """
    print("VALIDATING TEST DATA")
    print("=" * 25)
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    valid_pairs = []
    issues = []
    
    for file in os.listdir(test_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(test_dir, file)
            annotation_path = os.path.splitext(image_path)[0] + '.txt'
            
            if os.path.exists(annotation_path):
                # Validate annotation format
                try:
                    with open(annotation_path, 'r') as f:
                        line_count = 0
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:5]]
                                    
                                    # Validate ranges
                                    if class_id not in [0, 1, 2]:
                                        issues.append(f"{file}: Invalid class_id {class_id}")
                                    if not all(0.0 <= coord <= 1.0 for coord in coords):
                                        issues.append(f"{file}: Coordinates out of range [0,1]")
                                    
                                    line_count += 1
                                else:
                                    issues.append(f"{file}: Invalid annotation format in line: {line}")
                        
                        if line_count > 0:
                            valid_pairs.append((image_path, annotation_path))
                            print(f"✓ {file}: {line_count} annotations")
                        else:
                            issues.append(f"{file}: No valid annotations found")
                            
                except Exception as e:
                    issues.append(f"{file}: Error reading annotation file: {e}")
            else:
                issues.append(f"{file}: Missing annotation file {os.path.basename(annotation_path)}")
    
    print(f"\nFound {len(valid_pairs)} valid image-annotation pairs")
    
    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"❌ {issue}")
        return False
    else:
        print("✓ All test data is valid!")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Data Utility for Severity Calculator')
    parser.add_argument('--setup', action='store_true', help='Set up test directory')
    parser.add_argument('--validate', action='store_true', help='Validate test data')
    parser.add_argument('--visualize', type=str, help='Visualize annotations for image')
    parser.add_argument('--test_dir', type=str, 
                       default=r'C:\Users\rafab\OneDrive\Desktop\accuracy test',
                       help='Test directory path')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_test_directory(args.test_dir)
    elif args.validate:
        validate_test_data(args.test_dir)
    elif args.visualize:
        visualize_annotations(args.visualize)
    else:
        print("Use --setup, --validate, or --visualize <image_path>")
        print("Run with --help for more options")
