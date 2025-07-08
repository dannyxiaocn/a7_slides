import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

def create_collection_display(color_groups: Dict[str, List[str]], 
                            results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                            output_resolution: Tuple[int, int] = (1200, 800),
                            background_color: Tuple[int, int, int] = (240, 248, 255),  # Alice Blue
                            background_type: str = "solid") -> np.ndarray:
    """
    Create a collection display showing all butterfly groups with a single background.
    
    Args:
        color_groups: Dictionary mapping group names to lists of image filenames
        results: Dictionary mapping image names to (processed_image, mask) tuples
        output_resolution: (width, height) of final image in pixels
        background_color: RGB color for background
        background_type: "solid", "gradient", or "textured"
        
    Returns:
        Final collection display image as numpy array
    """
    width, height = output_resolution
    
    # Create background
    if background_type == "gradient":
        background = create_gradient_background(width, height, background_color)
    elif background_type == "textured":
        background = create_textured_background(width, height, background_color)
    else:  # solid
        background = create_solid_background(width, height, background_color)
    
    # Calculate layout
    num_groups = len(color_groups)
    layout = calculate_optimal_layout(color_groups, width, height)
    
    print(f"Creating collection display with layout: {layout}")
    print(f"Output resolution: {width}x{height}")
    print(f"Background: {background_type} ({background_color})")
    
    # Place butterflies on background
    final_image = place_butterflies_on_background(
        background, color_groups, results, layout, width, height
    )
    
    return final_image

def create_solid_background(width: int, height: int, 
                          color: Tuple[int, int, int]) -> np.ndarray:
    """Create a solid color background."""
    background = np.full((height, width, 3), color, dtype=np.uint8)
    return background

def create_gradient_background(width: int, height: int, 
                             base_color: Tuple[int, int, int]) -> np.ndarray:
    """Create a gradient background from dark to light."""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create vertical gradient
    for y in range(height):
        # Progress from 0.6 to 1.0 (dark to light)
        progress = 0.6 + 0.4 * (y / height)
        current_color = tuple(int(c * progress) for c in base_color)
        background[y, :] = current_color
    
    return background

def create_textured_background(width: int, height: int, 
                             base_color: Tuple[int, int, int]) -> np.ndarray:
    """Create a subtle textured background."""
    background = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add subtle noise texture
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add subtle geometric pattern
    for y in range(0, height, 40):
        for x in range(0, width, 40):
            if (x // 40 + y // 40) % 2 == 0:
                # Slightly lighter squares
                end_y = min(y + 40, height)
                end_x = min(x + 40, width)
                background[y:end_y, x:end_x] = np.clip(
                    background[y:end_y, x:end_x].astype(np.int16) + 5, 0, 255
                ).astype(np.uint8)
    
    return background

def calculate_optimal_layout(color_groups: Dict[str, List[str]], 
                           width: int, height: int) -> Dict[str, Dict]:
    """
    Calculate optimal layout for displaying butterfly groups.
    
    Returns:
        Dictionary with layout information for each group
    """
    num_groups = len(color_groups)
    
    # Calculate group layout (groups arranged vertically)
    group_height = height // num_groups
    
    layout = {}
    for i, (group_name, image_list) in enumerate(color_groups.items()):
        num_images = len(image_list)
        
        # Calculate images per row for this group
        images_per_row = math.ceil(math.sqrt(num_images))
        if images_per_row > 4:  # Max 4 per row for better visibility
            images_per_row = 4
        
        rows_needed = math.ceil(num_images / images_per_row)
        
        # Calculate dimensions for each butterfly in this group
        image_width = width // images_per_row
        image_height = group_height // max(rows_needed, 1)
        
        layout[group_name] = {
            "group_index": i,
            "y_start": i * group_height,
            "y_end": (i + 1) * group_height,
            "images_per_row": images_per_row,
            "rows": rows_needed,
            "image_width": image_width,
            "image_height": image_height,
            "num_images": num_images
        }
    
    return layout

def place_butterflies_on_background(background: np.ndarray,
                                   color_groups: Dict[str, List[str]],
                                   results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   layout: Dict[str, Dict],
                                   width: int, height: int) -> np.ndarray:
    """
    Place butterfly images on the background according to the layout.
    """
    final_image = background.copy()
    
    for group_name, image_list in color_groups.items():
        group_layout = layout[group_name]
        
        # Place butterflies (removed group title)
        for img_idx, image_name in enumerate(image_list):
            if image_name not in results:
                continue
                
            processed_image, mask = results[image_name]
            
            # Calculate position
            row = img_idx // group_layout["images_per_row"]
            col = img_idx % group_layout["images_per_row"]
            
            x_start = col * group_layout["image_width"]
            y_start = group_layout["y_start"] + row * group_layout["image_height"]  # Removed +50 for title
            
            # Resize and place butterfly
            resized_butterfly = resize_butterfly_for_display(
                processed_image, mask, 
                group_layout["image_width"] - 20,  # Leave margin
                group_layout["image_height"] - 20
            )
            
            # Center the butterfly in its allocated space
            butterfly_h, butterfly_w = resized_butterfly.shape[:2]
            center_x = x_start + (group_layout["image_width"] - butterfly_w) // 2
            center_y = y_start + (group_layout["image_height"] - butterfly_h) // 2
            
            # Place butterfly on final image
            final_image = blend_butterfly_onto_background(
                final_image, resized_butterfly, center_x, center_y
            )
            
            # Removed image name label
    
    return final_image

def resize_butterfly_for_display(processed_image: np.ndarray, 
                                mask: np.ndarray,
                                max_width: int, 
                                max_height: int) -> np.ndarray:
    """
    Resize butterfly image to fit within specified dimensions while maintaining aspect ratio.
    """
    # Convert BGRA to BGR for processing
    if processed_image.shape[2] == 4:
        bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
        alpha_channel = processed_image[:, :, 3]
    else:
        bgr_image = processed_image
        alpha_channel = mask
    
    # Find bounding box of the butterfly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # If no contours found, use the whole image
        x, y, w, h = 0, 0, bgr_image.shape[1], bgr_image.shape[0]
    else:
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop to bounding box with some padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(bgr_image.shape[1] - x, w + 2 * padding)
    h = min(bgr_image.shape[0] - y, h + 2 * padding)
    
    cropped_image = bgr_image[y:y+h, x:x+w]
    cropped_alpha = alpha_channel[y:y+h, x:x+w]
    
    # Calculate resize factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image and alpha
    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(cropped_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create BGRA image
    resized_bgra = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)
    resized_bgra[:, :, 3] = resized_alpha
    
    return resized_bgra

def blend_butterfly_onto_background(background: np.ndarray,
                                   butterfly: np.ndarray,
                                   x: int, y: int) -> np.ndarray:
    """
    Blend butterfly image onto background using alpha blending.
    """
    result = background.copy()
    
    # Get butterfly dimensions
    butterfly_h, butterfly_w = butterfly.shape[:2]
    
    # Ensure butterfly fits within background
    if x < 0 or y < 0 or x + butterfly_w > background.shape[1] or y + butterfly_h > background.shape[0]:
        return result
    
    # Extract alpha channel
    if butterfly.shape[2] == 4:
        alpha = butterfly[:, :, 3] / 255.0
        butterfly_bgr = butterfly[:, :, :3]
    else:
        alpha = np.ones((butterfly_h, butterfly_w), dtype=np.float32)
        butterfly_bgr = butterfly
    
    # Get background region
    bg_region = background[y:y+butterfly_h, x:x+butterfly_w]
    
    # Alpha blending
    for c in range(3):
        result[y:y+butterfly_h, x:x+butterfly_w, c] = (
            alpha * butterfly_bgr[:, :, c] + 
            (1 - alpha) * bg_region[:, :, c]
        )
    
    return result

def create_configurable_display(color_groups: Dict[str, List[str]], 
                               results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               images_per_class: Optional[int] = None,
                               output_resolution: Tuple[int, int] = (1200, 800),
                               background_style: str = "gradient") -> np.ndarray:
    """
    Create a configurable collection display with N images per class.
    
    Args:
        color_groups: Color group classifications
        results: Background removal results
        images_per_class: Number of images to show per class (None = all)
        output_resolution: Output size in pixels
        background_style: "solid", "gradient", or "textured"
    """
    # Limit images per class if specified
    if images_per_class is not None:
        limited_groups = {}
        for group_name, image_list in color_groups.items():
            limited_groups[group_name] = image_list[:images_per_class]
        color_groups = limited_groups
    
    # Choose background color based on style
    if background_style == "gradient":
        bg_color = (220, 235, 250)  # Light blue
    elif background_style == "textured":
        bg_color = (248, 248, 245)  # Warm white
    else:  # solid
        bg_color = (240, 248, 255)  # Alice blue
    
    return create_collection_display(
        color_groups, results, output_resolution, bg_color, background_style
    ) 