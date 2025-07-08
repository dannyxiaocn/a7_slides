import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
from background_removal import remove_background

def find_odd_by_color_distribution(color_groups: Dict[str, List[str]], 
                                 data_path: str = "../data/") -> Dict[str, str]:
    """
    Find the odd butterfly in each color group using ONLY color distribution analysis.
    IMPROVED VERSION: More sensitive to yellow+black vs pure yellow patterns.
    
    Args:
        color_groups: Dictionary mapping group names to lists of image filenames
        data_path: Path to the original images for background removal
        
    Returns:
        Dictionary mapping group names to the filename of the odd butterfly
    """
    odd_butterflies = {}
    
    print("Finding odd butterflies using IMPROVED COLOR DISTRIBUTION ANALYSIS...")
    print("="*70)
    
    for group_name, image_list in color_groups.items():
        if len(image_list) < 4:
            print(f"Group {group_name}: Not enough images ({len(image_list)}) to find odd one out")
            continue
            
        print(f"\nAnalyzing group: {group_name}")
        print(f"Images: {image_list}")
        
        # Extract color distribution features for all butterflies in this group
        color_features = {}
        for image_name in image_list:
            try:
                image_path = os.path.join(data_path, image_name)
                if os.path.exists(image_path):
                    print(f"  Processing {image_name} for color distribution...")
                    # Use background removal for clean color analysis
                    clean_image, clean_mask = remove_background(image_path, output_with_alpha=False)
                    color_dist = extract_improved_color_features(clean_image, clean_mask)
                    color_features[image_name] = color_dist
                else:
                    print(f"  Warning: Could not find {image_name}")
                    continue
                    
            except Exception as e:
                print(f"  Error processing {image_name}: {e}")
                continue
                
        if len(color_features) < 4:
            print(f"Not enough valid features extracted for group {group_name}")
            continue
            
        # Find the odd one out based on color distribution
        odd_butterfly = detect_improved_color_outlier(color_features, group_name)
        odd_butterflies[group_name] = odd_butterfly
        
        print(f"IMPROVED COLOR DISTRIBUTION ODD BUTTERFLY: {odd_butterfly}")
    
    return odd_butterflies

def extract_improved_color_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    IMPROVED color feature extraction with better sensitivity for yellow+black vs pure yellow.
    """
    # Convert to multiple color spaces for comprehensive analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgr = image
    
    # Extract only butterfly pixels
    butterfly_pixels_hsv = hsv[mask > 0]
    butterfly_pixels_bgr = bgr[mask > 0]
    
    if len(butterfly_pixels_hsv) == 0:
        return create_empty_improved_features()
    
    h, s, v = butterfly_pixels_hsv[:, 0], butterfly_pixels_hsv[:, 1], butterfly_pixels_hsv[:, 2]
    b, g, r = butterfly_pixels_bgr[:, 0], butterfly_pixels_bgr[:, 1], butterfly_pixels_bgr[:, 2]
    
    # IMPROVED Feature 1: Enhanced black detection (key for Group 2)
    black_score = improved_black_detection(h, s, v, b, g, r)
    
    # IMPROVED Feature 2: Enhanced yellow detection
    yellow_score = improved_yellow_detection(h, s, v)
    
    # IMPROVED Feature 3: Multi-color complexity (yellow+black vs pure yellow)
    multi_color_complexity = improved_multi_color_analysis(h, s, v)
    
    # IMPROVED Feature 4: Pattern contrast (sharp color transitions)
    pattern_contrast = analyze_pattern_contrast(image, mask)
    
    # IMPROVED Feature 5: Color distribution entropy
    color_entropy = improved_color_entropy(h, s, v)
    
    # IMPROVED Feature 6: Dark-light ratio analysis
    dark_light_ratio = analyze_dark_light_contrast(v)
    
    # IMPROVED Feature 7: Color boundary detection
    color_boundaries = detect_color_boundaries(image, mask)
    
    # IMPROVED Feature 8: Dual-tone analysis (specifically for yellow+black)
    dual_tone_score = analyze_dual_tone_patterns(h, s, v)
    
    return {
        "black_score": black_score,
        "yellow_score": yellow_score,
        "multi_color_complexity": multi_color_complexity,
        "pattern_contrast": pattern_contrast,
        "color_entropy": color_entropy,
        "dark_light_ratio": dark_light_ratio,
        "color_boundaries": color_boundaries,
        "dual_tone_score": dual_tone_score
    }

def improved_black_detection(h: np.ndarray, s: np.ndarray, v: np.ndarray, 
                           b: np.ndarray, g: np.ndarray, r: np.ndarray) -> float:
    """SUPER AGGRESSIVE black detection - prioritizing any significant black presence."""
    total_pixels = len(v)
    if total_pixels == 0:
        return 0.0
    
    # Method 1: Very aggressive low value detection
    very_dark = (v < 40).sum()      # Increased threshold
    moderately_dark = (v < 80).sum() # Even broader dark detection
    
    # Method 2: Aggressive RGB black detection
    rgb_very_dark = ((r < 60) & (g < 60) & (b < 60)).sum()  # Increased threshold
    rgb_dark = ((r < 80) & (g < 80) & (b < 80)).sum()       # Broader detection
    
    # Method 3: Any dark pixels regardless of saturation
    any_dark = (v < 50).sum()
    
    # Method 4: Look for dark edges/patterns (brown-black, dark orange)
    dark_colored = ((h >= 0) & (h <= 30) & (v < 100)).sum()  # Dark browns/oranges
    
    # Method 5: Conservative true black
    true_black = ((r < 40) & (g < 40) & (b < 40)).sum()
    
    # Take the MAXIMUM detection from all methods (most aggressive)
    black_candidates = [very_dark, moderately_dark, rgb_very_dark, rgb_dark, 
                       any_dark, dark_colored, true_black]
    black_pixels = max(black_candidates)
    black_ratio = float(black_pixels) / total_pixels
    
    # SUPER AGGRESSIVE scoring: massively amplify ANY black presence
    if black_ratio > 0.25:  # Very high black presence
        return black_ratio * 5.0  # Huge amplification
    elif black_ratio > 0.15:  # High black presence  
        return black_ratio * 4.0  # Strong amplification
    elif black_ratio > 0.08:  # Medium black presence
        return black_ratio * 3.0  # Medium amplification
    elif black_ratio > 0.03:  # Any detectable black
        return black_ratio * 2.5  # Still amplify small amounts
    else:
        return black_ratio

def improved_yellow_detection(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
    """IMPROVED yellow detection with better hue range."""
    # Expanded yellow range in HSV (OpenCV uses 0-180 for hue)
    # Yellow can range from 15-45 degrees
    yellow_mask1 = (h >= 15) & (h <= 45) & (s > 40) & (v > 80)  # Bright yellow
    yellow_mask2 = (h >= 10) & (h <= 50) & (s > 20) & (v > 50)  # Broader yellow
    
    yellow_pixels = max(yellow_mask1.sum(), yellow_mask2.sum())
    total_pixels = len(h)
    
    yellow_ratio = float(yellow_pixels) / total_pixels if total_pixels > 0 else 0.0
    return yellow_ratio

def improved_multi_color_analysis(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
    """IMPROVED multi-color analysis focusing on yellow+black combinations."""
    # Define precise color bins
    color_bins = {
        'black': ((v < 50)),
        'dark_brown': ((h >= 0) & (h < 15) & (s > 30) & (v >= 50) & (v < 100)),
        'yellow': ((h >= 15) & (h <= 45) & (s > 40) & (v > 80)),
        'orange': ((h >= 5) & (h < 25) & (s > 50) & (v > 100)),
        'light': ((v > 150))
    }
    
    significant_colors = 0
    color_counts = {}
    
    total_pixels = len(h)
    for color_name, mask in color_bins.items():
        count = mask.sum()
        ratio = count / total_pixels if total_pixels > 0 else 0
        color_counts[color_name] = ratio
        
        if ratio > 0.08:  # At least 8% for significant presence
            significant_colors += 1
    
    # Special scoring for yellow+black combination (key for Group 2)
    yellow_black_combo = 0.0
    if color_counts['yellow'] > 0.2 and color_counts['black'] > 0.1:
        yellow_black_combo = 2.0  # High score for yellow+black combination
    
    complexity_score = float(significant_colors) + yellow_black_combo
    return complexity_score

def analyze_pattern_contrast(image: np.ndarray, mask: np.ndarray) -> float:
    """Analyze contrast patterns within the butterfly."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    butterfly_gray = gray * (mask > 0).astype(np.uint8)
    
    if not np.any(mask > 0):
        return 0.0
    
    # Calculate local contrast using Sobel operator
    grad_x = cv2.Sobel(butterfly_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(butterfly_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Average gradient in butterfly region
    mask_area = mask > 0
    contrast = float(np.mean(gradient_magnitude[mask_area])) if np.any(mask_area) else 0.0
    
    return contrast

def improved_color_entropy(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
    """IMPROVED color entropy calculation."""
    # Filter meaningful color pixels
    meaningful_pixels = (s > 15) & (v > 30)
    
    if meaningful_pixels.sum() < 20:
        return 0.0
    
    valid_h = h[meaningful_pixels]
    
    # Use more bins for better entropy calculation
    hist, _ = np.histogram(valid_h, bins=36, range=(0, 180))  # 5-degree bins
    
    # Normalize to probabilities
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    
    hist = hist / total
    
    # Calculate entropy
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def analyze_dark_light_contrast(v: np.ndarray) -> float:
    """Analyze contrast between dark and light regions."""
    if len(v) == 0:
        return 0.0
    
    # Calculate the range and standard deviation of values
    value_range = float(v.max() - v.min())
    value_std = float(np.std(v))
    
    # Combine range and std for contrast measure
    contrast = (value_range + value_std * 2) / 3.0
    
    return contrast

def detect_color_boundaries(image: np.ndarray, mask: np.ndarray) -> float:
    """Detect sharp color boundaries (edges between different colors)."""
    # Convert to HSV for color edge detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect edges in hue channel (color changes)
    hue_edges = cv2.Canny((h * (mask > 0).astype(np.uint8)).astype(np.uint8), 30, 100)
    
    # Count edge pixels in butterfly region
    edge_pixels = np.sum(hue_edges > 0)
    total_pixels = np.sum(mask > 0)
    
    boundary_density = float(edge_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
    
    return boundary_density

def analyze_dual_tone_patterns(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
    """SUPER AGGRESSIVE dual-tone analysis focusing on ANY yellow+black combination."""
    # More aggressive yellow detection (broader range)
    yellow_pixels_broad = ((h >= 10) & (h <= 50) & (s > 20) & (v > 50)).sum()
    yellow_pixels_strict = ((h >= 15) & (h <= 45) & (s > 40) & (v > 80)).sum()
    yellow_pixels = max(yellow_pixels_broad, yellow_pixels_strict)
    
    # More aggressive black detection (multiple methods)
    black_pixels_value = (v < 50).sum()
    black_pixels_rgb = ((h >= 0) & (h <= 30) & (v < 80)).sum()  # Include dark browns
    black_pixels = max(black_pixels_value, black_pixels_rgb)
    
    total_pixels = len(h)
    if total_pixels == 0:
        return 0.0
    
    yellow_ratio = yellow_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    
    # SUPER AGGRESSIVE dual-tone scoring
    dual_tone_score = 0.0
    
    # Even more aggressive scoring for yellow+black combinations
    if yellow_ratio > 0.15 and black_ratio > 0.08:
        dual_tone_score = 5.0  # MAXIMUM score for any significant dual-tone
    elif yellow_ratio > 0.10 and black_ratio > 0.05:
        dual_tone_score = 4.0  # High score for moderate dual-tone
    elif yellow_ratio > 0.05 and black_ratio > 0.03:
        dual_tone_score = 3.0  # Medium score for weak dual-tone
    elif yellow_ratio > 0.3 and black_ratio < 0.02:
        dual_tone_score = 0.2  # Very low score for pure yellow
    else:
        dual_tone_score = yellow_ratio * black_ratio * 10.0  # Multiplicative bonus
    
    return dual_tone_score

def create_empty_improved_features() -> Dict[str, float]:
    """Create empty feature dict for error cases."""
    return {
        "black_score": 0.0,
        "yellow_score": 0.0,
        "multi_color_complexity": 0.0,
        "pattern_contrast": 0.0,
        "color_entropy": 0.0,
        "dark_light_ratio": 0.0,
        "color_boundaries": 0.0,
        "dual_tone_score": 0.0
    }

def detect_improved_color_outlier(features: Dict[str, Dict[str, float]], group_name: str) -> str:
    """
    IMPROVED outlier detection focusing on yellow+black vs pure yellow patterns.
    """
    image_names = list(features.keys())
    n_images = len(image_names)
    
    if n_images < 4:
        return image_names[0]
    
    print(f"  IMPROVED Color Distribution Analysis for {n_images} butterflies...")
    
    # Print detailed analysis for each butterfly
    for img_name in image_names:
        feat = features[img_name]
        print(f"    {img_name}:")
        print(f"      Black score: {feat['black_score']:.3f}")
        print(f"      Yellow score: {feat['yellow_score']:.3f}")
        print(f"      Multi-color complexity: {feat['multi_color_complexity']:.2f}")
        print(f"      Dual-tone score: {feat['dual_tone_score']:.2f}")
        print(f"      Color entropy: {feat['color_entropy']:.2f}")
        print(f"      Pattern contrast: {feat['pattern_contrast']:.1f}")
    
    # IMPROVED: Focus on features that distinguish yellow+black from pure yellow
    key_features = [
        "dual_tone_score",       # MOST IMPORTANT: yellow+black vs pure yellow
        "black_score",           # Black presence
        "multi_color_complexity", # Multiple colors
        "color_boundaries",      # Sharp color transitions
        "pattern_contrast"       # Visual contrast
    ]
    
    # SUPER AGGRESSIVE weights prioritizing black detection
    feature_weights = {
        "black_score": 5.0,          # MAXIMUM weight for black presence detection
        "dual_tone_score": 4.5,      # Very high weight for yellow+black detection
        "multi_color_complexity": 3.0,
        "color_boundaries": 2.0,
        "pattern_contrast": 1.5,
        "dark_light_ratio": 1.2,     # Increased for dark-light contrast
        "color_entropy": 1.0,
        "yellow_score": 0.2          # Very low weight for yellow (all have yellow)
    }
    
    # Calculate weighted distances
    avg_distances = {}
    
    for img1 in image_names:
        distances = []
        for img2 in image_names:
            if img1 != img2:
                dist = calculate_improved_color_distance(features[img1], features[img2], feature_weights)
                distances.append(dist)
        avg_distances[img1] = np.mean(distances)
    
    print(f"  IMPROVED color distribution distances:")
    for img in sorted(avg_distances.keys()):
        dist = avg_distances[img]
        print(f"    {img}: avg distance = {dist:.3f}")
    
    # The odd one out has the highest average distance
    odd_butterfly = max(avg_distances.items(), key=lambda x: x[1])[0]
    print(f"  -> IMPROVED COLOR DISTRIBUTION ODD ONE: {odd_butterfly}")
    
    return odd_butterfly

def calculate_improved_color_distance(features1: Dict[str, float], features2: Dict[str, float], 
                                    feature_weights: Dict[str, float]) -> float:
    """Calculate improved weighted distance between two color feature vectors."""
    weighted_diffs = []
    
    for feature_name, weight in feature_weights.items():
        val1 = features1.get(feature_name, 0)
        val2 = features2.get(feature_name, 0)
        
        # Normalize by the maximum value to handle different scales
        max_val = max(abs(val1), abs(val2), 1e-6)
        normalized_diff = abs(val1 - val2) / max_val
        
        # Apply weight
        weighted_diff = weight * (normalized_diff ** 2)
        weighted_diffs.append(weighted_diff)
    
    return np.sqrt(sum(weighted_diffs))