import cv2
import numpy as np
from typing import Tuple, Optional
import warnings

def remove_background(image_path: str, output_with_alpha: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from butterfly images using advanced classical image processing techniques.
    
    Args:
        image_path: Path to input image
        output_with_alpha: If True, returns RGBA image with transparent background
        
    Returns:
        Tuple of (processed_image, mask) where:
        - processed_image: Image with background removed (RGBA if output_with_alpha=True)
        - mask: Binary mask showing foreground regions
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original_image = image.copy()
    height, width = image.shape[:2]
    
    # Try multiple methods in order of sophistication
    methods = [
        ('grabcut_auto', apply_grabcut_auto),
        ('grabcut_center', apply_grabcut_center_bias),
        ('advanced_color', apply_advanced_color_segmentation),
        ('watershed', apply_watershed_segmentation),
        ('combined_fallback', apply_combined_fallback)
    ]
    
    best_mask = None
    best_score = 0
    
    for method_name, method_func in methods:
        try:
            mask = method_func(image)
            score = evaluate_mask_quality(mask, image)
            
            print(f"Method {method_name}: quality score = {score:.3f}")
            
            if score > best_score:
                best_mask = mask
                best_score = score
                
            # If we get a good enough result, use it
            if score > 0.7:
                print(f"Using {method_name} (score: {score:.3f})")
                break
                
        except Exception as e:
            print(f"Method {method_name} failed: {e}")
            continue
    
    if best_mask is None:
        # Last resort: simple thresholding
        best_mask = create_simple_mask(image)
        print("Using simple fallback mask")
    
    # Post-process the mask
    final_mask = post_process_mask(best_mask, image)
    
    # Create final result
    if output_with_alpha:
        result = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = final_mask
    else:
        result = cv2.bitwise_and(original_image, original_image, mask=final_mask)
    
    return result, final_mask

def apply_grabcut_auto(image: np.ndarray) -> np.ndarray:
    """Apply GrabCut with automatic rectangle initialization."""
    height, width = image.shape[:2]
    
    # Create initial rectangle (avoid edges where background typically is)
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
    
    # Initialize masks
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    return mask2 * 255

def apply_grabcut_center_bias(image: np.ndarray) -> np.ndarray:
    """Apply GrabCut with center-biased initialization."""
    height, width = image.shape[:2]
    
    # Create a mask that assumes center region is likely foreground
    mask = np.zeros((height, width), np.uint8)
    
    # Mark edges as probable background
    edge_thickness = min(width, height) // 8
    mask[:edge_thickness, :] = cv2.GC_BGD  # Top edge
    mask[-edge_thickness:, :] = cv2.GC_BGD  # Bottom edge
    mask[:, :edge_thickness] = cv2.GC_BGD  # Left edge
    mask[:, -edge_thickness:] = cv2.GC_BGD  # Right edge
    
    # Mark center region as probable foreground
    center_x, center_y = width // 2, height // 2
    center_size = min(width, height) // 3
    y1 = max(0, center_y - center_size // 2)
    y2 = min(height, center_y + center_size // 2)
    x1 = max(0, center_x - center_size // 2)
    x2 = min(width, center_x + center_size // 2)
    mask[y1:y2, x1:x2] = cv2.GC_FGD
    
    # Everything else is probable background
    mask[mask == 0] = cv2.GC_PR_BGD
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    rect = (0, 0, 0, 0)  # Dummy rect since we're using mask initialization
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
    
    # Create binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    return mask2 * 255

def apply_advanced_color_segmentation(image: np.ndarray) -> np.ndarray:
    """Apply advanced color-based segmentation using multiple color spaces."""
    height, width = image.shape[:2]
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # HSV-based mask
    h, s, v = cv2.split(hsv)
    
    # Adaptive thresholding for saturation
    s_thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5)
    
    # Value thresholding to remove very dark areas
    _, v_thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # LAB-based mask focusing on A and B channels (color information)
    l, a, b = cv2.split(lab)
    
    # Enhance color channels
    a_enhanced = cv2.equalizeHist(a)
    b_enhanced = cv2.equalizeHist(b)
    
    # Combine color information
    color_mask = cv2.bitwise_and(s_thresh, v_thresh)
    
    # Add edge information
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(color_mask, edges_dilated)
    
    # Fill holes
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(combined_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (height * width * 0.001):  # Filter small noise
            cv2.fillPoly(filled_mask, [contour], (255,))
    
    return filled_mask

def apply_watershed_segmentation(image: np.ndarray) -> np.ndarray:
    """Apply watershed segmentation for background removal."""
    height, width = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Threshold to create binary image
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg_uint8 = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg_uint8)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg_uint8)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create mask from watershed result
    mask = np.zeros_like(gray)
    mask[markers > 1] = 255
    
    return mask

def apply_combined_fallback(image: np.ndarray) -> np.ndarray:
    """Fallback method combining multiple simple techniques."""
    height, width = image.shape[:2]
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: Color-based thresholding
    h, s, v = cv2.split(hsv)
    _, s_thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
    _, v_thresh = cv2.threshold(v, 30, 255, cv2.THRESH_BINARY)
    color_mask = cv2.bitwise_and(s_thresh, v_thresh)
    
    # Method 3: Edge-based
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Combine methods
    combined = cv2.bitwise_or(adaptive_thresh, color_mask)
    combined = cv2.bitwise_or(combined, edges_closed)
    
    return combined

def create_simple_mask(image: np.ndarray) -> np.ndarray:
    """Create a simple mask as last resort."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def evaluate_mask_quality(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Evaluate the quality of a segmentation mask.
    Returns a score between 0 and 1, where higher is better.
    """
    height, width = image.shape[:2]
    total_pixels = height * width
    
    # Basic checks
    foreground_pixels = np.sum(mask > 0)
    background_pixels = total_pixels - foreground_pixels
    
    # Avoid masks that are too empty or too full
    fg_ratio = foreground_pixels / total_pixels
    if fg_ratio < 0.05 or fg_ratio > 0.95:
        return 0.0
    
    # Check for connectivity (prefer fewer, larger connected components)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_components = len(contours)
    
    if num_components == 0:
        return 0.0
    
    # Calculate largest component ratio
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    largest_ratio = largest_area / total_pixels
    
    # Score based on:
    # 1. Reasonable foreground ratio (0.1-0.8 is good)
    # 2. Not too many small components
    # 3. Largest component should be significant
    
    ratio_score = 1.0 - abs(fg_ratio - 0.4) * 2  # Peak at 0.4, penalty for extremes
    ratio_score = max(0, ratio_score)
    
    component_score = 1.0 / (1.0 + num_components * 0.1)  # Penalty for many components
    
    largest_score = min(1.0, largest_ratio * 3)  # Reward for having a large main component
    
    total_score = (ratio_score + component_score + largest_score) / 3
    
    return total_score

def post_process_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Post-process the mask to improve quality."""
    # Remove small noise with opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small holes with closing
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
    
    # Keep only significant contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    # Filter contours by area
    height, width = mask.shape
    total_area = height * width
    min_area = total_area * 0.001
    
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.fillPoly(filtered_mask, [contour], (255,))
    
    # Final smoothing
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel_final)
    
    return final_mask

def process_all_images(data_path: str, output_dir: Optional[str] = None) -> dict:
    """
    Process all images in the data directory for background removal.
    
    Args:
        data_path: Path to directory containing images
        output_dir: Optional output directory for processed images
        
    Returns:
        Dictionary mapping image names to (processed_image, mask) tuples
    """
    import os
    
    results = {}
    
    # Get all image files
    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(data_path, image_file)
        try:
            print(f"\nProcessing {image_file}...")
            processed_image, mask = remove_background(image_path)
            results[image_file] = (processed_image, mask)
            
            # Save processed images if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                # Save RGBA image
                output_path = os.path.join(output_dir, f"processed_{image_file}")
                cv2.imwrite(output_path, processed_image)
                # Save mask
                mask_path = os.path.join(output_dir, f"mask_{image_file}")
                cv2.imwrite(mask_path, mask)
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results
