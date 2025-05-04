import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes, dilation
from skimage.filters import threshold_niblack
from skimage.segmentation import slic

def apply_variance_thresholding(image, window_size=25, k=-0.2):
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

def overlay_cracks_only(original_image, binary_crack_mask, alpha=0.5):
    mask = (binary_crack_mask == 255).astype(np.uint8)  # Ensure mask is uint8
    overlay = np.zeros_like(original_image)
    overlay[:, :] = [0, 0, 255]  # Blue
    result = original_image.copy()
    result[mask == 1] = cv2.addWeighted(original_image[mask == 1], 1 - alpha, overlay[mask == 1], alpha, 0)
    return result

def mask_outside_bbox_with_superpixels(image, bbox_color=(0, 255, 0), n_segments=400, overlap_threshold=0.99):
    # Find green bounding box with more precise color range
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the green mask
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    coords = cv2.findNonZero(green_mask)
    if coords is None or len(coords) < 4:
        print("Warning: No green bounding box found. Using entire image.")
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Create a binary mask for the bounding box
    bbox_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bbox_mask[y:y+h, x:x+w] = 255

    # Generate superpixels
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=1)
    
    # Create visualization of superpixels with boundaries
    superpixel_viz = image.copy()
    for segment in np.unique(segments):
        mask = segments == segment
        # Find boundaries of the segment
        boundaries = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        # Draw white boundaries
        superpixel_viz[boundaries == 1] = [255, 255, 255]
    
    # Create the final mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for seg_val in np.unique(segments):
        seg_mask = segments == seg_val
        # Calculate the percentage of the superpixel that overlaps with the bounding box
        overlap = np.sum(seg_mask & (bbox_mask == 255)) / np.sum(seg_mask)
        # More strict condition: must be well inside the box and not touching the edges
        if overlap >= overlap_threshold and np.all(seg_mask[green_mask > 0] == 0):
            mask[seg_mask] = 255

    # Apply additional morphological operations to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove any remaining content outside the bounding box
    mask = cv2.bitwise_and(mask, bbox_mask)
    
    # Create a black background
    result = np.zeros_like(image)
    # Apply the mask to the original image
    result[mask == 255] = image[mask == 255]
    
    # Extremely aggressive removal of the box lines, especially at the top and left
    # 1. Create a protective mask for the inside of the box
    protect_mask = np.ones_like(mask)
    top_padding = 40  # Extremely aggressive padding for top
    left_padding = 30  # Extremely aggressive padding for left
    right_padding = 8  # Less aggressive padding for right
    bottom_padding = 5  # Less aggressive padding for bottom
    cv2.rectangle(protect_mask, 
                 (x+left_padding, y+top_padding), 
                 (x+w-right_padding, y+h-bottom_padding), 1, -1)
    
    # 2. Create a mask for the box lines and their immediate surroundings
    line_mask = np.zeros_like(mask)
    # Very thick lines for top and left, thinner for bottom and right
    cv2.rectangle(line_mask, (x, y), (x+w, y+h), 255, 15)  # Extremely thick lines overall
    # Dilate the line mask to include more surrounding area
    line_mask = cv2.dilate(line_mask, kernel, iterations=8)  # Many more dilation iterations
    
    # 3. Remove green pixels with a very wide range but only from the box lines
    hsv_result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    # Extremely aggressive green detection for top and left, less for bottom and right
    green_pixels = cv2.inRange(hsv_result, (5, 5, 5), (120, 255, 255))  # Extremely wide range
    # Only remove green pixels that are part of the box lines and not protected
    green_pixels = cv2.bitwise_and(green_pixels, line_mask)
    green_pixels = cv2.bitwise_and(green_pixels, cv2.bitwise_not(protect_mask))
    result[green_pixels > 0] = 0
    
    # 4. Additional pass to remove any remaining green tint near the lines
    near_lines = cv2.dilate(line_mask, kernel, iterations=8)  # Many more dilation iterations
    near_lines = cv2.bitwise_and(near_lines, cv2.bitwise_not(protect_mask))
    # Extremely aggressive green detection for near-line areas
    hsv_near = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    green_near = cv2.inRange(hsv_near, (5, 5, 5), (120, 255, 255))  # Extremely wide range
    result[np.logical_and(near_lines > 0, green_near > 0)] = 0
    
    # 5. Final cleanup of any remaining artifacts
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(result, result, mask=binary)
    
    return result, mask, superpixel_viz

def reduce_noise(mask, kernel_size=3, min_size=50):
    """Reduce noise in a binary mask using morphological operations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Convert to boolean for remove_small_objects
    mask_bool = mask.astype(bool)
    # Remove small objects
    mask_bool = remove_small_objects(mask_bool, min_size=min_size)
    # Remove small holes
    mask_bool = remove_small_holes(mask_bool, area_threshold=min_size)
    # Convert back to uint8 for morphological operations
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    # Apply morphological closing to fill small gaps
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    # Apply morphological opening to remove small noise
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    return mask_uint8

# ---- MAIN ----
if __name__ == "__main__":
    input_path = "C:/Users/bentl/Desktop/CSPROJECT/Alligator-Crack_0.29_20250430_073258_GPS_placeholder.jpg"
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 360))

    # 1. Apply superpixels and remove everything outside green bounding box
    cleaned_image, inside_mask, superpixel_viz = mask_outside_bbox_with_superpixels(image)

    # 2. Apply crack detection
    var_mask = apply_variance_thresholding(cleaned_image)
    nib_mask = apply_niblack_thresholding(cleaned_image)

    # 3. Combine threshold outputs
    combined_mask = cv2.bitwise_or(var_mask, nib_mask)
    
    # 4. Reduce noise in the combined mask
    combined_mask_clean = reduce_noise(combined_mask, kernel_size=5, min_size=100)

    # 5. Calculate crack area
    # Convert mask to binary (0 or 1)
    crack_mask = (combined_mask_clean == 255).astype(np.uint8)
    # Calculate total number of crack pixels
    total_crack_pixels = np.sum(crack_mask)
    # Calculate area in pixels
    crack_area_pixels = total_crack_pixels
    # Calculate area percentage of the total image
    total_image_pixels = image.shape[0] * image.shape[1]
    crack_area_percentage = (crack_area_pixels / total_image_pixels) * 100

    # 6. Apply overlay based on white areas
    overlay = overlay_cracks_only(cleaned_image, combined_mask)
    overlay_clean = overlay_cracks_only(cleaned_image, combined_mask_clean)

    # ---- Display ----
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.ravel()

    # First row
    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(superpixel_viz)
    axes[1].set_title("Superpixel Segmentation")

    axes[2].imshow(cleaned_image)
    axes[2].set_title("Masked Inside Bounding Box")

    axes[3].imshow(overlay)
    axes[3].set_title("Original Crack Overlay")

    # Second row
    axes[4].imshow(var_mask, cmap='gray')
    axes[4].set_title("Variance Threshold")

    axes[5].imshow(nib_mask, cmap='gray')
    axes[5].set_title("Niblack Threshold")

    axes[6].imshow(combined_mask_clean, cmap='gray')
    axes[6].set_title(f"Noise-Reduced Combined Mask\nCrack Area: {crack_area_pixels} pixels\n({crack_area_percentage:.2f}% of image)")

    axes[7].imshow(overlay_clean)
    axes[7].set_title("Noise-Reduced Crack Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Print crack area information
    print(f"\nCrack Analysis:")
    print(f"Total crack pixels: {crack_area_pixels}")
    print(f"Crack area percentage: {crack_area_percentage:.2f}%")
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
