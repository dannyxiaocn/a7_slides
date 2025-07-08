import cv2
import numpy as np
from typing import List, Tuple, Dict
import math
import os
from background_removal import remove_background

def find_odd_butterflies(color_groups: Dict[str, List[str]], 
                        results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                        data_path: str = "../data/") -> Dict[str, str]:
    """
    Find the odd butterfly in each color group using classical image processing features.
    Enhanced with background removal for better feature extraction.
    
    Args:
        color_groups: Dictionary mapping group names to lists of image filenames
        results: Dictionary mapping image names to (processed_image, mask) tuples
        data_path: Path to the original images for background removal
        
    Returns:
        Dictionary mapping group names to the filename of the odd butterfly
    """
    odd_butterflies = {}
    
    print("Finding odd butterflies in each color group...")
    print("Using enhanced background removal for better feature extraction...")
    print("="*60)
    
    for group_name, image_list in color_groups.items():
        if len(image_list) < 4:
            print(f"Group {group_name}: Not enough images ({len(image_list)}) to find odd one out")
            continue
            
        print(f"\nAnalyzing group: {group_name}")
        print(f"Images: {image_list}")
        
        # Extract features for all butterflies in this group using enhanced background removal
        features = {}
        for image_name in image_list:
            try:
                # Get better background removal for feature extraction
                image_path = os.path.join(data_path, image_name)
                if os.path.exists(image_path):
                    print(f"  Processing {image_name} with enhanced background removal...")
                    enhanced_image, enhanced_mask = remove_background(image_path, output_with_alpha=False)
                    feature_vector = extract_butterfly_features(enhanced_image, enhanced_mask)
                elif image_name in results:
                    # Fallback to existing results
                    processed_image, mask = results[image_name]
                    feature_vector = extract_butterfly_features(processed_image, mask)
                else:
                    print(f"  Warning: Could not find {image_name}")
                    continue
                    
                features[image_name] = feature_vector
                
            except Exception as e:
                print(f"  Error processing {image_name}: {e}")
                # Fallback to existing results if available
                if image_name in results:
                    processed_image, mask = results[image_name]
                    feature_vector = extract_butterfly_features(processed_image, mask)
                    features[image_name] = feature_vector
                
        if len(features) < 4:
            print(f"Not enough valid features extracted for group {group_name}")
            continue
            
        # Find the odd one out
        odd_butterfly = detect_odd_one_out(features, group_name)
        odd_butterflies[group_name] = odd_butterfly
        
        print(f"Odd butterfly detected: {odd_butterfly}")
    
    return odd_butterflies

def extract_butterfly_features(processed_image: np.ndarray, 
                              mask: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive features from a butterfly image using classical methods.
    Enhanced for better human-like visual pattern recognition with cleaner background removal.
    
    Features extracted:
    - Enhanced pattern features: wing patterns, color distribution, visual complexity
    - Shape features: area, perimeter, aspect ratio, solidity, extent
    - Color features: dominant colors, color variance, distribution
    - Texture features: contrast, energy, homogeneity
    - Geometric features: convex hull properties, moments
    - Wing structure features: bilateral symmetry, wing span analysis
    """
    # Convert BGRA to BGR for processing
    if processed_image.shape[2] == 4:
        bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = processed_image
    
    # Extract enhanced pattern features (most important for species differentiation)
    enhanced_pattern_features = extract_enhanced_pattern_features(bgr_image, mask)
    
    # Extract color distribution features
    color_features = extract_color_distribution_features(bgr_image, mask)
    
    # Extract shape features
    shape_features = extract_shape_features(mask)
    
    # Extract texture features
    texture_features = extract_texture_features(bgr_image, mask)
    
    # Extract basic pattern features
    pattern_features = extract_pattern_features(bgr_image, mask)
    
    # Extract geometric features
    geometric_features = extract_geometric_features(mask)
    
    # Extract new wing structure features for better species differentiation
    wing_features = extract_wing_structure_features(bgr_image, mask)
    
    # Extract visual complexity features that are key for distinguishing species
    complexity_features = extract_visual_complexity_features(bgr_image, mask)
    
    # Combine all features with enhanced patterns having priority
    all_features = {
        **enhanced_pattern_features, 
        **wing_features,
        **complexity_features,
        **color_features,
        **shape_features, 
        **texture_features, 
        **pattern_features, 
        **geometric_features
    }
    
    return all_features

def extract_enhanced_pattern_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract enhanced pattern features that better capture visual differences humans notice."""
    # Convert to different color spaces for pattern analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply mask to focus on butterfly
    masked_gray = cv2.bitwise_and(gray, mask)
    
    if not np.any(mask > 0):
        return {"spot_density": 0, "line_patterns": 0, "color_transitions": 0, "wing_pattern_variance": 0}
    
    # 1. Spot/Circle detection (eye spots, markings)
    spot_density = detect_circular_patterns(masked_gray, mask)
    
    # 2. Line pattern detection (stripes, veining)
    line_patterns = detect_line_patterns(masked_gray, mask)
    
    # 3. Color transition analysis (gradients, bands)
    color_transitions = analyze_color_transitions(image, mask)
    
    # 4. Wing pattern variance (complexity of patterns)
    wing_pattern_variance = analyze_wing_pattern_variance(masked_gray, mask)
    
    return {
        "spot_density": spot_density,
        "line_patterns": line_patterns,
        "color_transitions": color_transitions,
        "wing_pattern_variance": wing_pattern_variance
    }

def detect_circular_patterns(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Detect circular patterns like eye spots."""
    # Use HoughCircles to detect circular patterns
    circles = cv2.HoughCircles(
        gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )
    
    if circles is None:
        return 0.0
    
    # Count circles that are within the butterfly mask
    valid_circles = 0
    circles = np.round(circles[0, :]).astype("int")
    
    for (x, y, r) in circles:
        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
            valid_circles += 1
    
    # Normalize by mask area
    mask_area = np.sum(mask > 0)
    return float(valid_circles) / (mask_area / 10000.0) if mask_area > 0 else 0.0

def detect_line_patterns(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Detect linear patterns like stripes."""
    # Use HoughLines to detect linear patterns
    edges = cv2.Canny(gray_image, 50, 150)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLines(masked_edges, 1, np.pi/180, threshold=30)
    
    if lines is None:
        return 0.0
    
    # Count significant lines
    line_count = len(lines)
    
    # Normalize by mask area
    mask_area = np.sum(mask > 0)
    return float(line_count) / (mask_area / 10000.0) if mask_area > 0 else 0.0

def analyze_color_transitions(image: np.ndarray, mask: np.ndarray) -> float:
    """Analyze color transitions and gradients."""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply mask
    h_masked = cv2.bitwise_and(h, mask)
    s_masked = cv2.bitwise_and(s, mask)
    
    if not np.any(mask > 0):
        return 0.0
    
    # Calculate gradients in hue and saturation
    h_grad_x = cv2.Sobel(h_masked, cv2.CV_64F, 1, 0, ksize=3)
    h_grad_y = cv2.Sobel(h_masked, cv2.CV_64F, 0, 1, ksize=3)
    
    s_grad_x = cv2.Sobel(s_masked, cv2.CV_64F, 1, 0, ksize=3)
    s_grad_y = cv2.Sobel(s_masked, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    h_gradient = np.sqrt(h_grad_x**2 + h_grad_y**2)
    s_gradient = np.sqrt(s_grad_x**2 + s_grad_y**2)
    
    # Average gradient in butterfly region
    h_transitions = float(np.mean(h_gradient[mask > 0])) if np.any(mask > 0) else 0.0
    s_transitions = float(np.mean(s_gradient[mask > 0])) if np.any(mask > 0) else 0.0
    
    return (h_transitions + s_transitions) / 2.0

def analyze_wing_pattern_variance(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Analyze variance in wing patterns."""
    if not np.any(mask > 0):
        return 0.0
    
    # Calculate local variance using a sliding window
    kernel = np.ones((5, 5), np.float32) / 25
    gray_float = gray_image.astype(np.float32)
    local_mean = cv2.filter2D(gray_float, cv2.CV_32F, kernel)
    variance_image = (gray_float - local_mean) ** 2
    local_variance = cv2.filter2D(variance_image, cv2.CV_32F, kernel)
    
    # Average variance in butterfly region
    variance = float(np.mean(local_variance[mask > 0])) if np.any(mask > 0) else 0.0
    
    return variance

def extract_color_distribution_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract color distribution features that humans notice."""
    if not np.any(mask > 0):
        return {"color_variance": 0, "dominant_color_strength": 0, "color_uniformity": 0}
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Get butterfly pixels
    h_pixels = h[mask > 0]
    s_pixels = s[mask > 0]
    v_pixels = v[mask > 0]
    
    # Color variance (how varied are the colors)
    color_variance = float(np.var(h_pixels.astype(np.float64))) if len(h_pixels) > 0 else 0.0
    
    # Dominant color strength (how concentrated is the main color)
    h_hist, _ = np.histogram(h_pixels, bins=180, range=(0, 180))
    h_hist = h_hist.astype(np.float32)
    h_hist /= (np.sum(h_hist) + 1e-7)
    dominant_color_strength = float(np.max(h_hist))
    
    # Color uniformity (inverse of color variance)
    color_uniformity = 1.0 / (1.0 + color_variance / 100.0)
    
    return {
        "color_variance": color_variance,
        "dominant_color_strength": dominant_color_strength,
        "color_uniformity": color_uniformity
    }

def extract_shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Extract shape-based features from the butterfly mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 1, "solidity": 0, "extent": 0}
    
    # Get the largest contour (main butterfly)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Basic shape measurements
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent
    }

def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract texture features using Gray Level Co-occurrence Matrix properties."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply mask to focus on butterfly region
    masked_gray = cv2.bitwise_and(gray, mask)
    
    # Get butterfly region only
    butterfly_pixels = masked_gray[mask > 0]
    
    if len(butterfly_pixels) == 0:
        return {"contrast": 0, "energy": 0, "homogeneity": 0, "entropy": 0}
    
    # Calculate basic texture statistics
    mean_intensity = float(np.mean(butterfly_pixels.astype(np.float64)))
    std_intensity = float(np.std(butterfly_pixels.astype(np.float64)))
    
    # Calculate histogram-based features
    hist, _ = np.histogram(butterfly_pixels, bins=32, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (np.sum(hist) + 1e-7)  # Normalize
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    
    # Calculate contrast using Laplacian
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    laplacian_values = laplacian[mask > 0]
    contrast = float(np.var(laplacian_values.astype(np.float64))) if np.any(mask > 0) else 0.0
    
    # Energy (uniformity)
    energy = np.sum(hist ** 2)
    
    # Homogeneity (inverse difference moment)
    homogeneity = calculate_homogeneity(masked_gray, mask)
    
    return {
        "contrast": float(contrast),
        "energy": float(energy),
        "homogeneity": float(homogeneity),
        "entropy": float(entropy),
        "intensity_mean": float(mean_intensity),
        "intensity_std": float(std_intensity)
    }

def calculate_homogeneity(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate homogeneity using a simplified approach."""
    if not np.any(mask > 0):
        return 0.0
    
    # Get butterfly region
    butterfly_region = gray_image[mask > 0]
    
    # Calculate local variance as a measure of homogeneity
    # Higher variance = lower homogeneity
    if len(butterfly_region) < 2:
        return 1.0
    
    local_variance = float(np.var(butterfly_region.astype(np.float64)))
    # Convert to homogeneity (inverse relationship)
    homogeneity = 1.0 / (1.0 + local_variance / 100.0)
    
    return homogeneity

def extract_pattern_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract pattern and edge-based features."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Edge density
    total_mask_pixels = np.sum(mask > 0)
    edge_pixels = np.sum(masked_edges > 0)
    edge_density = float(edge_pixels) / total_mask_pixels if total_mask_pixels > 0 else 0
    
    # Symmetry analysis (simplified)
    symmetry_score = calculate_symmetry(mask)
    
    # Pattern complexity (based on contour hierarchy)
    complexity = calculate_pattern_complexity(masked_edges)
    
    return {
        "edge_density": edge_density,
        "symmetry": symmetry_score,
        "pattern_complexity": complexity
    }

def calculate_symmetry(mask: np.ndarray) -> float:
    """Calculate bilateral symmetry score."""
    if not np.any(mask > 0):
        return 0.0
    
    height, width = mask.shape
    
    # Find the center of mass
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return 0.0
    
    cx = int(moments['m10'] / moments['m00'])
    
    # Compare left and right halves
    left_half = mask[:, :cx]
    right_half = mask[:, cx:]
    
    # Flip right half for comparison
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize to same width for comparison
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_resized = left_half[:, -min_width:] if left_half.shape[1] > min_width else left_half
    right_resized = right_half_flipped[:, :min_width] if right_half_flipped.shape[1] > min_width else right_half_flipped
    
    # Calculate similarity
    if left_resized.shape != right_resized.shape:
        return 0.0
    
    intersection = np.sum((left_resized > 0) & (right_resized > 0))
    union = np.sum((left_resized > 0) | (right_resized > 0))
    
    symmetry = float(intersection) / union if union > 0 else 0.0
    
    return symmetry

def calculate_pattern_complexity(edges: np.ndarray) -> float:
    """Calculate pattern complexity based on edge distribution."""
    if not np.any(edges > 0):
        return 0.0
    
    # Find contours in edge image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Count internal contours (holes, patterns)
    internal_contours = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] != -1:  # Has parent (internal contour)
                internal_contours += 1
    
    # Normalize by total contours
    complexity = float(internal_contours) / len(contours) if len(contours) > 0 else 0.0
    
    return complexity

def extract_geometric_features(mask: np.ndarray) -> Dict[str, float]:
    """Extract geometric features using image moments."""
    moments = cv2.moments(mask)
    
    if moments['m00'] == 0:
        return {"hu_moment_1": 0, "hu_moment_2": 0, "hu_moment_3": 0, "compactness": 0}
    
    # Hu moments (scale, rotation, translation invariant)
    hu_moments = cv2.HuMoments(moments)
    
    # Use first few Hu moments (log transform for stability)
    hu1 = -np.sign(hu_moments[0]) * np.log10(abs(hu_moments[0]) + 1e-10)
    hu2 = -np.sign(hu_moments[1]) * np.log10(abs(hu_moments[1]) + 1e-10)
    hu3 = -np.sign(hu_moments[2]) * np.log10(abs(hu_moments[2]) + 1e-10)
    
    # Compactness (circularity measure)
    area = moments['m00']
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    return {
        "hu_moment_1": float(hu1),
        "hu_moment_2": float(hu2), 
        "hu_moment_3": float(hu3),
        "compactness": float(compactness)
    }

def detect_odd_one_out(features: Dict[str, Dict[str, float]], group_name: str) -> str:
    """
    Detect the odd butterfly using enhanced weighted distance analysis.
    Prioritizes visual pattern features that humans notice most.
    """
    image_names = list(features.keys())
    n_images = len(image_names)
    
    if n_images < 4:
        return image_names[0]  # Default fallback
    
    print(f"  Analyzing {n_images} butterflies for oddness with enhanced pattern focus...")
    
    # Define feature weights (higher = more important for species differentiation)
    # Enhanced weights for better Group 2 detection while maintaining performance on Groups 1&3
    feature_weights = {
        # Enhanced pattern features (most important for humans)
        "spot_density": 4.5,
        "line_patterns": 4.5,
        "color_transitions": 4.0,
        "wing_pattern_variance": 3.5,
        
        # New wing structure features (very important for species differentiation)
        "bilateral_symmetry": 3.8,
        "wing_span_ratio": 3.2,
        "wing_area_ratio": 2.8,
        "central_body_ratio": 2.5,
        
        # Visual complexity features (key for distinguishing species)
        "pattern_density": 4.0,
        "color_complexity": 3.5,
        "texture_entropy": 3.0,
        "edge_complexity": 3.2,
        
        # Color distribution features (important)
        "color_variance": 2.8,
        "dominant_color_strength": 2.2,
        "color_uniformity": 1.8,
        
        # Shape features (moderately important)
        "area": 1.2,
        "aspect_ratio": 2.0,
        "solidity": 1.5,
        "extent": 1.2,
        
        # Pattern features
        "edge_density": 1.8,
        "symmetry": 1.5,
        "pattern_complexity": 2.2,
        
        # Texture features (less important for species)
        "contrast": 0.8,
        "energy": 0.8,
        "homogeneity": 0.8,
        "entropy": 0.8,
        "intensity_mean": 0.6,
        "intensity_std": 0.6,
        
        # Geometric features (least important)
        "hu_moment_1": 0.4,
        "hu_moment_2": 0.4,
        "hu_moment_3": 0.4,
        "compactness": 0.6
    }
    
    # Calculate weighted distances
    distances = calculate_enhanced_distances(features, image_names, feature_weights)
    
    # Calculate multiple distance metrics for robustness
    euclidean_avg = {}
    manhattan_avg = {}
    pattern_focused_avg = {}
    
    for img in image_names:
        other_distances = [distances[img][other] for other in image_names if other != img]
        euclidean_avg[img] = np.mean(other_distances)
        
        # Manhattan distance (more robust to outliers)
        manhattan_distances = [calculate_manhattan_distance(features[img], features[other], feature_weights) 
                             for other in image_names if other != img]
        manhattan_avg[img] = np.mean(manhattan_distances)
        
        # Pattern-focused distance (only enhanced pattern features)
        pattern_distances = [calculate_pattern_focused_distance(features[img], features[other]) 
                           for other in image_names if other != img]
        pattern_focused_avg[img] = np.mean(pattern_distances)
    
    # Combine different distance measures with weights
    combined_scores = {}
    for img in image_names:
        combined_scores[img] = (
            0.3 * euclidean_avg[img] +     # Standard weighted distance  
            0.2 * manhattan_avg[img] +     # Robust distance
            0.5 * pattern_focused_avg[img] # Pattern-focused (most important)
        )
    
    print(f"    Enhanced analysis results:")
    print(f"    Pattern-focused weighting: spots/lines/transitions prioritized")
    for img in sorted(combined_scores.keys()):
        score = combined_scores[img]
        print(f"      {img}: combined score = {score:.3f}")
    
    # The odd one out is the one with the highest combined score
    odd_butterfly = max(combined_scores.items(), key=lambda x: x[1])[0]
    print(f"    -> ODD ONE OUT: {odd_butterfly}")
    
    return odd_butterfly

def calculate_feature_distance(features1: Dict[str, float], 
                             features2: Dict[str, float],
                             feature_names: List[str]) -> float:
    """
    Calculate normalized Euclidean distance between two feature vectors.
    """
    distances = []
    
    for feature_name in feature_names:
        val1 = features1.get(feature_name, 0)
        val2 = features2.get(feature_name, 0)
        
        # Normalize the difference by the maximum value to handle different scales
        max_val = max(abs(val1), abs(val2), 1e-10)
        normalized_diff = abs(val1 - val2) / max_val
        distances.append(normalized_diff)
    
    # Return Euclidean distance
    return math.sqrt(sum(d ** 2 for d in distances))

def calculate_enhanced_distances(features: Dict[str, Dict[str, float]], 
                               image_names: List[str], 
                               feature_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Calculate weighted Euclidean distances between all pairs of butterflies."""
    distances = {}
    
    for i, img1 in enumerate(image_names):
        distances[img1] = {}
        for j, img2 in enumerate(image_names):
            if i != j:
                dist = calculate_weighted_euclidean_distance(
                    features[img1], features[img2], feature_weights
                )
                distances[img1][img2] = dist
    
    return distances

def calculate_weighted_euclidean_distance(features1: Dict[str, float], 
                                        features2: Dict[str, float],
                                        feature_weights: Dict[str, float]) -> float:
    """Calculate weighted Euclidean distance between two feature vectors."""
    weighted_squared_diffs = []
    
    for feature_name, weight in feature_weights.items():
        val1 = features1.get(feature_name, 0)
        val2 = features2.get(feature_name, 0)
        
        # Normalize the difference
        max_val = max(abs(val1), abs(val2), 1e-10)
        normalized_diff = abs(val1 - val2) / max_val
        
        # Apply weight
        weighted_squared_diff = weight * (normalized_diff ** 2)
        weighted_squared_diffs.append(weighted_squared_diff)
    
    return math.sqrt(sum(weighted_squared_diffs))

def calculate_manhattan_distance(features1: Dict[str, float], 
                               features2: Dict[str, float],
                               feature_weights: Dict[str, float]) -> float:
    """Calculate weighted Manhattan distance between two feature vectors."""
    weighted_diffs = []
    
    for feature_name, weight in feature_weights.items():
        val1 = features1.get(feature_name, 0)
        val2 = features2.get(feature_name, 0)
        
        # Normalize the difference
        max_val = max(abs(val1), abs(val2), 1e-10)
        normalized_diff = abs(val1 - val2) / max_val
        
        # Apply weight
        weighted_diff = weight * normalized_diff
        weighted_diffs.append(weighted_diff)
    
    return sum(weighted_diffs)

def calculate_pattern_focused_distance(features1: Dict[str, float], 
                                     features2: Dict[str, float]) -> float:
    """Calculate distance using only the most important pattern and structure features."""
    # Enhanced pattern features focusing on species-differentiating characteristics
    pattern_features = [
        "spot_density", "line_patterns", "color_transitions", 
        "wing_pattern_variance", "color_variance", "pattern_complexity",
        "bilateral_symmetry", "wing_span_ratio", "wing_area_ratio",
        "pattern_density", "color_complexity", "edge_complexity"
    ]
    
    pattern_weights = {
        "spot_density": 1.0,
        "line_patterns": 1.0,
        "color_transitions": 0.9,
        "wing_pattern_variance": 0.8,
        "color_variance": 0.7,
        "pattern_complexity": 0.7,
        "bilateral_symmetry": 0.9,
        "wing_span_ratio": 0.8,
        "wing_area_ratio": 0.7,
        "pattern_density": 0.9,
        "color_complexity": 0.8,
        "edge_complexity": 0.8
    }
    
    weighted_squared_diffs = []
    
    for feature_name in pattern_features:
        if feature_name in pattern_weights:
            val1 = features1.get(feature_name, 0)
            val2 = features2.get(feature_name, 0)
            
            # Normalize the difference
            max_val = max(abs(val1), abs(val2), 1e-10)
            normalized_diff = abs(val1 - val2) / max_val
            
            # Apply pattern weight
            weight = pattern_weights[feature_name]
            weighted_squared_diff = weight * (normalized_diff ** 2)
            weighted_squared_diffs.append(weighted_squared_diff)
    
    return math.sqrt(sum(weighted_squared_diffs)) if weighted_squared_diffs else 0.0

def visualize_odd_detection_results(color_groups: Dict[str, List[str]],
                                   results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   odd_butterflies: Dict[str, str]) -> None:
    """
    Visualize the odd one out detection results.
    """
    import matplotlib.pyplot as plt
    
    num_groups = len(color_groups)
    fig, axes = plt.subplots(num_groups, 4, figsize=(16, 4*num_groups))
    
    if num_groups == 1:
        axes = axes.reshape(1, -1)
    
    for group_idx, (group_name, image_list) in enumerate(color_groups.items()):
        odd_butterfly = odd_butterflies.get(group_name, "")
        
        for img_idx, image_name in enumerate(image_list[:4]):  # Show max 4 images
            if image_name in results:
                processed_image, _ = results[image_name]
                display_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGB)
                
                axes[group_idx, img_idx].imshow(display_image)
                
                # Highlight the odd one
                title = image_name.replace('.jpg', '')
                if image_name == odd_butterfly:
                    title += " (ODD)"
                    axes[group_idx, img_idx].set_title(title, fontsize=10, 
                                                      color='red', fontweight='bold')
                    # Add red border
                    for spine in axes[group_idx, img_idx].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                else:
                    axes[group_idx, img_idx].set_title(title, fontsize=10)
                
                axes[group_idx, img_idx].axis('off')
        
        # Add group label
        axes[group_idx, 0].text(-0.1, 0.5, group_name, 
                               transform=axes[group_idx, 0].transAxes,
                               rotation=90, verticalalignment='center',
                               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def extract_wing_structure_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract wing structure features that help distinguish butterfly species.
    Focus on bilateral symmetry, wing span analysis, and structural patterns.
    """
    if not np.any(mask > 0):
        return {"bilateral_symmetry": 0, "wing_span_ratio": 0, "wing_area_ratio": 0, "central_body_ratio": 0}
    
    # Find the main contour of the butterfly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"bilateral_symmetry": 0, "wing_span_ratio": 0, "wing_area_ratio": 0, "central_body_ratio": 0}
    
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate bilateral symmetry
    bilateral_symmetry = calculate_bilateral_symmetry(mask, main_contour)
    
    # Calculate wing span ratio (width to height)
    x, y, w, h = cv2.boundingRect(main_contour)
    wing_span_ratio = float(w) / float(h) if h > 0 else 0.0
    
    # Calculate wing area distribution
    wing_area_ratio = calculate_wing_area_distribution(mask, main_contour)
    
    # Calculate central body ratio
    central_body_ratio = calculate_central_body_ratio(mask, main_contour)
    
    return {
        "bilateral_symmetry": bilateral_symmetry,
        "wing_span_ratio": wing_span_ratio,
        "wing_area_ratio": wing_area_ratio,
        "central_body_ratio": central_body_ratio
    }

def calculate_bilateral_symmetry(mask: np.ndarray, contour: np.ndarray) -> float:
    """Calculate how bilaterally symmetric the butterfly is."""
    # Find the center of the butterfly
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Create left and right halves
    h, w = mask.shape
    left_mask = mask.copy()
    right_mask = mask.copy()
    
    left_mask[:, cx:] = 0  # Keep only left half
    right_mask[:, :cx] = 0  # Keep only right half
    
    # Flip the right half to compare with left
    right_flipped = cv2.flip(right_mask, 1)
    
    # Calculate overlap between left half and flipped right half
    overlap = cv2.bitwise_and(left_mask, right_flipped)
    overlap_area = np.sum(overlap > 0)
    total_area = max(int(np.sum(left_mask > 0)), int(np.sum(right_mask > 0)), 1)
    
    symmetry_score = float(overlap_area) / float(total_area)
    return symmetry_score

def calculate_wing_area_distribution(mask: np.ndarray, contour: np.ndarray) -> float:
    """Calculate the distribution of wing area."""
    # Find the bounding box and divide into quadrants
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create quadrant masks
    mid_x = x + w // 2
    mid_y = y + h // 2
    
    q1 = mask[y:mid_y, x:mid_x]  # Top-left
    q2 = mask[y:mid_y, mid_x:x+w]  # Top-right
    q3 = mask[mid_y:y+h, x:mid_x]  # Bottom-left
    q4 = mask[mid_y:y+h, mid_x:x+w]  # Bottom-right
    
    areas = [np.sum(q > 0) for q in [q1, q2, q3, q4]]
    total_area = sum(areas)
    
    if total_area == 0:
        return 0.0
    
    # Calculate the variance in quadrant areas (normalized)
    area_ratios = [area / total_area for area in areas]
    mean_ratio = np.mean(area_ratios)
    variance = np.mean([(ratio - mean_ratio) ** 2 for ratio in area_ratios])
    
    return float(variance)

def calculate_central_body_ratio(mask: np.ndarray, contour: np.ndarray) -> float:
    """Calculate the ratio of central body area to total butterfly area."""
    # Find the center and create a central region
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Create a central circular region (body area)
    h, w = mask.shape
    body_mask = np.zeros_like(mask)
    
    # Estimate body radius as a fraction of the butterfly size
    x, y, bw, bh = cv2.boundingRect(contour)
    body_radius = min(bw, bh) // 6  # Body is roughly 1/6 of the butterfly size
    
    cv2.circle(body_mask, (cx, cy), body_radius, (255,), -1)
    
    # Calculate overlap between body mask and butterfly mask
    body_overlap = cv2.bitwise_and(mask, body_mask)
    body_area = np.sum(body_overlap > 0)
    total_area = np.sum(mask > 0)
    
    return float(body_area) / float(total_area) if total_area > 0 else 0.0

def extract_visual_complexity_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract visual complexity features that help distinguish species.
    These capture the overall visual 'business' and pattern complexity.
    """
    if not np.any(mask > 0):
        return {"pattern_density": 0, "color_complexity": 0, "texture_entropy": 0, "edge_complexity": 0}
    
    # Convert to grayscale for some analyses
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, mask)
    
    # Pattern density - how many distinct patterns/features
    pattern_density = calculate_pattern_density(masked_gray, mask)
    
    # Color complexity - variance in color space
    color_complexity = calculate_color_complexity(image, mask)
    
    # Texture entropy - randomness of texture
    texture_entropy = calculate_texture_entropy(masked_gray, mask)
    
    # Edge complexity - complexity of edge patterns
    edge_complexity = calculate_edge_complexity(masked_gray, mask)
    
    return {
        "pattern_density": pattern_density,
        "color_complexity": color_complexity,
        "texture_entropy": texture_entropy,
        "edge_complexity": edge_complexity
    }

def calculate_pattern_density(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate the density of distinct patterns in the image."""
    # Use corner detection to find distinct features
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=mask)
    
    if corners is None:
        return 0.0
    
    # Normalize by area
    mask_area = np.sum(mask > 0)
    pattern_density = float(len(corners)) / (mask_area / 1000.0) if mask_area > 0 else 0.0
    
    return pattern_density

def calculate_color_complexity(image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate complexity in the color distribution."""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract colors only from butterfly region
    butterfly_pixels = hsv[mask > 0]
    
    if len(butterfly_pixels) == 0:
        return 0.0
    
    # Calculate variance in hue and saturation
    hue_var = float(np.var(butterfly_pixels[:, 0]))
    sat_var = float(np.var(butterfly_pixels[:, 1]))
    
    # Combine variances
    color_complexity = (hue_var + sat_var) / 2.0
    
    return color_complexity

def calculate_texture_entropy(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate texture entropy (randomness) in the butterfly region."""
    # Extract pixels from butterfly region
    butterfly_pixels = gray_image[mask > 0]
    
    if len(butterfly_pixels) == 0:
        return 0.0
    
    # Calculate histogram
    hist, _ = np.histogram(butterfly_pixels, bins=32, range=(0, 256))
    
    # Normalize histogram to get probabilities
    hist = hist.astype(float)
    hist /= np.sum(hist)
    
    # Calculate entropy
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_edge_complexity(gray_image: np.ndarray, mask: np.ndarray) -> float:
    """Calculate complexity of edge patterns."""
    # Detect edges
    edges = cv2.Canny(gray_image, 50, 150)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Count edge pixels
    edge_pixels = np.sum(masked_edges > 0)
    total_pixels = np.sum(mask > 0)
    
    if total_pixels == 0:
        return 0.0
    
    # Calculate edge density
    edge_density = float(edge_pixels) / float(total_pixels)
    
    # Calculate edge curvature/complexity
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_curvature = 0.0
    for contour in contours:
        if len(contour) > 10:  # Need enough points to calculate curvature
            # Approximate contour to get curvature measure
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # More points in approximation = higher curvature
            curvature = float(len(contour)) / float(max(len(approx), 1))
            total_curvature += curvature
    
    # Combine edge density and curvature
    edge_complexity = edge_density * (1.0 + total_curvature / 100.0)
    
    return edge_complexity 