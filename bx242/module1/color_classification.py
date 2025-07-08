import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_butterfly_hue_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract HUE-based color features from a butterfly image using its mask.
    
    Args:
        image: Original BGR image
        mask: Binary mask showing butterfly regions
        
    Returns:
        Dictionary containing HUE statistics
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply mask to get only butterfly pixels
    butterfly_pixels_mask = mask > 0
    
    if not np.any(butterfly_pixels_mask):
        return {"mean_hue": 0, "median_hue": 0, "dominant_hue": 0, "hue_std": 0}
    
    # Extract HUE values only from butterfly regions
    butterfly_hues = h[butterfly_pixels_mask]
    butterfly_saturations = s[butterfly_pixels_mask]
    butterfly_values = v[butterfly_pixels_mask]
    
    # Filter out low saturation pixels (they don't have meaningful color information)
    high_sat_mask = butterfly_saturations > 30  # Threshold for meaningful color
    
    if np.any(high_sat_mask):
        meaningful_hues = butterfly_hues[high_sat_mask]
    else:
        meaningful_hues = butterfly_hues
    
    # Calculate HUE statistics
    mean_hue = float(np.mean(meaningful_hues.astype(np.float64)))
    median_hue = float(np.median(meaningful_hues.astype(np.float64)))
    hue_std = float(np.std(meaningful_hues.astype(np.float64)))
    
    # Find dominant HUE using histogram
    hue_hist, bins = np.histogram(meaningful_hues, bins=180, range=(0, 180))
    dominant_hue = np.argmax(hue_hist)
    
    return {
        "mean_hue": float(mean_hue),
        "median_hue": float(median_hue),
        "dominant_hue": float(dominant_hue),
        "hue_std": float(hue_std),
        "num_pixels": len(meaningful_hues)
    }

def classify_butterflies_by_color(results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                  target_groups: int = 3, 
                                  images_per_group: int = 4) -> Dict[str, List[str]]:
    """
    Classify butterfly images into color groups based on HUE analysis.
    
    Args:
        results: Dictionary mapping image names to (processed_image, mask) tuples
        target_groups: Number of color groups to create
        images_per_group: Expected number of images per group
        
    Returns:
        Dictionary mapping group names to lists of image filenames
    """
    # Extract HUE features for all images
    image_features = {}
    image_names = list(results.keys())
    
    print("Extracting HUE features from butterfly images...")
    for image_name in image_names:
        processed_image, mask = results[image_name]
        # Convert BGRA back to BGR for analysis
        bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
        features = extract_butterfly_hue_features(bgr_image, mask)
        image_features[image_name] = features
        print(f"{image_name}: Mean HUE = {features['mean_hue']:.1f}°, "
              f"Dominant HUE = {features['dominant_hue']:.1f}°")
    
    # Use dominant HUE as the primary classification feature
    hue_values = [features['dominant_hue'] for features in image_features.values()]
    
    # Perform classical HUE-based grouping (NO machine learning)
    clusters = perform_classical_hue_grouping(hue_values, image_names, target_groups)
    
    # Ensure each cluster has exactly the target number of images
    balanced_clusters = balance_groups(clusters, images_per_group)
    
    # Create color group names based on HUE ranges
    color_groups = {}
    hue_to_color_name = get_hue_color_mapping()
    
    for i, (group_name, image_list) in enumerate(balanced_clusters.items()):
        # Calculate average HUE for this group
        group_hues = [image_features[img]['dominant_hue'] for img in image_list]
        avg_hue = np.mean(group_hues)
        color_name = hue_to_color_name(float(avg_hue))
        
        color_groups[f"Group_{i+1}_{color_name}"] = image_list
        
        print(f"\n{color_name} Group (Average HUE: {avg_hue:.1f}°):")
        for img in image_list:
            hue = image_features[img]['dominant_hue']
            print(f"  - {img}: {hue:.1f}°")
    
    return color_groups

def perform_classical_hue_grouping(hue_values: List[float], 
                                  image_names: List[str], 
                                  num_groups: int) -> Dict[str, List[str]]:
    """
    Perform classical HUE-based grouping using fixed thresholds (NO machine learning).
    Uses predefined color wheel divisions to assign butterflies to color groups.
    """
    # Create list of (hue, name) pairs
    hue_image_pairs = list(zip(hue_values, image_names))
    
    # Sort by HUE values to understand distribution
    sorted_pairs = sorted(hue_image_pairs, key=lambda x: x[0])
    sorted_hues = [pair[0] for pair in sorted_pairs]
    
    print(f"HUE distribution: min={min(sorted_hues):.1f}°, max={max(sorted_hues):.1f}°")
    
    # Method 1: Try fixed color wheel divisions first
    groups = try_fixed_color_divisions(sorted_pairs, num_groups)
    
    # Method 2: If fixed divisions don't work well, use simple quantile-based division
    if not is_good_grouping(groups):
        print("Fixed color divisions didn't work well, using quantile-based grouping...")
        groups = simple_quantile_grouping(sorted_pairs, num_groups)
    
    return groups

def try_fixed_color_divisions(sorted_pairs: List[Tuple[float, str]], 
                             num_groups: int) -> Dict[str, List[str]]:
    """
    Try to group butterflies using fixed color wheel divisions.
    """
    if num_groups != 3:
        # For non-3 groups, fall back to simple division
        return simple_quantile_grouping(sorted_pairs, num_groups)
    
    # Define three main color regions on the color wheel
    # Red: 0-60°, Green: 60-120°, Blue: 120-180°
    red_group = []
    green_group = []
    blue_group = []
    
    for hue, name in sorted_pairs:
        if hue <= 60:
            red_group.append(name)
        elif hue <= 120:
            green_group.append(name)
        else:
            blue_group.append(name)
    
    return {
        "group_0": red_group,
        "group_1": green_group, 
        "group_2": blue_group
    }

def simple_quantile_grouping(sorted_pairs: List[Tuple[float, str]], 
                           num_groups: int) -> Dict[str, List[str]]:
    """
    Simple quantile-based grouping - divide sorted HUE values into equal parts.
    This is purely mathematical division, not machine learning.
    """
    total_images = len(sorted_pairs)
    images_per_group = total_images // num_groups
    
    groups = {}
    for i in range(num_groups):
        start_idx = i * images_per_group
        if i == num_groups - 1:  # Last group gets any remaining images
            end_idx = total_images
        else:
            end_idx = start_idx + images_per_group
        
        group_pairs = sorted_pairs[start_idx:end_idx]
        group_names = [pair[1] for pair in group_pairs]
        groups[f"group_{i}"] = group_names
        
        # Print group info
        group_hues = [pair[0] for pair in group_pairs]
        print(f"Group {i}: HUE range {min(group_hues):.1f}°-{max(group_hues):.1f}°, {len(group_names)} images")
    
    return groups

def is_good_grouping(groups: Dict[str, List[str]]) -> bool:
    """
    Check if the grouping is reasonably balanced.
    """
    group_sizes = [len(group) for group in groups.values()]
    min_size = min(group_sizes)
    max_size = max(group_sizes)
    
    # Consider it good if the difference between largest and smallest group is <= 2
    return (max_size - min_size) <= 2

def balance_groups(clusters: Dict[str, List[str]], 
                  target_size: int) -> Dict[str, List[str]]:
    """
    Balance clusters to have exactly the target size by reassigning images.
    """
    all_images = []
    cluster_centers = []
    
    # Flatten all images and calculate cluster centers
    for cluster_name, image_list in clusters.items():
        all_images.extend(image_list)
        # Use cluster index as a simple center representation
        cluster_id = int(cluster_name.split('_')[1])
        cluster_centers.append(cluster_id)
    
    # If we already have the right distribution, return as is
    if all(len(img_list) == target_size for img_list in clusters.values()):
        return clusters
    
    # Simple redistribution: sort all images by their current cluster assignment
    # and redistribute them evenly
    sorted_clusters = sorted(clusters.items())
    balanced_clusters = {}
    
    all_sorted_images = []
    for _, image_list in sorted_clusters:
        all_sorted_images.extend(image_list)
    
    # Redistribute into balanced groups
    for i in range(len(sorted_clusters)):
        start_idx = i * target_size
        end_idx = start_idx + target_size
        balanced_clusters[f"cluster_{i}"] = all_sorted_images[start_idx:end_idx]
    
    return balanced_clusters

def get_hue_color_mapping():
    """
    Map HUE values to color names.
    HUE ranges: Red(0-30,150-180), Orange(30-45), Yellow(45-75), 
    Green(75-105), Blue(105-135), Purple(135-150)
    """
    def map_hue_to_color(hue: float) -> str:
        hue = hue % 180  # Ensure within range
        
        if hue < 15 or hue > 165:
            return "Red"
        elif 15 <= hue < 30:
            return "Red-Orange"
        elif 30 <= hue < 45:
            return "Orange"
        elif 45 <= hue < 60:
            return "Yellow-Orange"
        elif 60 <= hue < 75:
            return "Yellow"
        elif 75 <= hue < 90:
            return "Yellow-Green"
        elif 90 <= hue < 105:
            return "Green"
        elif 105 <= hue < 120:
            return "Blue-Green"
        elif 120 <= hue < 135:
            return "Blue"
        elif 135 <= hue < 150:
            return "Blue-Purple"
        elif 150 <= hue < 165:
            return "Purple"
        else:
            return "Red-Purple"
    
    return map_hue_to_color

def visualize_color_classification(results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                 color_groups: Dict[str, List[str]]) -> None:
    """
    Visualize the color classification results.
    """
    num_groups = len(color_groups)
    fig, axes = plt.subplots(num_groups, 4, figsize=(16, 4*num_groups))
    
    if num_groups == 1:
        axes = axes.reshape(1, -1)
    
    for group_idx, (group_name, image_list) in enumerate(color_groups.items()):
        for img_idx, image_name in enumerate(image_list):
            if img_idx < 4:  # Ensure we don't exceed 4 images per group
                # Load original image for display
                processed_image, mask = results[image_name]
                
                # Convert BGRA to RGB for display
                display_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGB)
                
                axes[group_idx, img_idx].imshow(display_image)
                axes[group_idx, img_idx].set_title(f'{image_name}', fontsize=10)
                axes[group_idx, img_idx].axis('off')
        
        # Add group label
        axes[group_idx, 0].text(-0.1, 0.5, group_name, 
                               transform=axes[group_idx, 0].transAxes,
                               rotation=90, verticalalignment='center',
                               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def analyze_hue_distribution(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """
    Analyze and visualize the HUE distribution of all butterflies.
    """
    all_hues = []
    image_hue_data = {}
    
    print("Analyzing HUE distribution...")
    
    for image_name, (processed_image, mask) in results.items():
        bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
        features = extract_butterfly_hue_features(bgr_image, mask)
        all_hues.append(features['dominant_hue'])
        image_hue_data[image_name] = features['dominant_hue']
    
    # Create HUE distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of HUE values
    ax1.hist(all_hues, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('HUE Value (degrees)')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Distribution of Dominant HUE Values')
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot showing each image's HUE
    sorted_items = sorted(image_hue_data.items(), key=lambda x: x[1])
    x_pos = range(len(sorted_items))
    hue_values = [item[1] for item in sorted_items]
    image_names = [item[0] for item in sorted_items]
    
    scatter = ax2.scatter(x_pos, hue_values, c=hue_values, cmap='hsv', s=100, alpha=0.8)
    ax2.set_xlabel('Image Index (sorted by HUE)')
    ax2.set_ylabel('HUE Value (degrees)')
    ax2.set_title('HUE Values by Image')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace('.jpg', '') for name in image_names], 
                       rotation=45, ha='right')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label='HUE Value')
    
    plt.tight_layout()
    plt.show()
    
    return image_hue_data 