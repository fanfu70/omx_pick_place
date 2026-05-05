import numpy as np
import cv2

# HSV ranges for common colors (OpenCV uses H: 0-179)
COLOR_THRESHOLDS = {
    'red': {'lower': np.array([0, 50, 50]), 'upper': np.array([15, 255, 255])},
    'red2': {'lower': np.array([160, 50, 50]), 'upper': np.array([180, 255, 255])}, # Red wraps around
    'green': {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
    'blue': {'lower': np.array([100, 50, 50]), 'upper': np.array([140, 255, 255])},
    'yellow': {'lower': np.array([20, 50, 50]), 'upper': np.array([35, 255, 255])},
}

def get_color_mask(hsv_image, color_name='red'):
    """Get a binary mask for the specified color."""
    mask = np.zeros_like(hsv_image[:, :, 0])
    
    # Handle red special case (wraps around 180)
    if color_name == 'red':
        mask1 = cv2.inRange(hsv_image, COLOR_THRESHOLDS['red']['lower'], COLOR_THRESHOLDS['red']['upper'])
        mask2 = cv2.inRange(hsv_image, COLOR_THRESHOLDS['red2']['lower'], COLOR_THRESHOLDS['red2']['upper'])
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_name in COLOR_THRESHOLDS:
        mask = cv2.inRange(hsv_image, COLOR_THRESHOLDS[color_name]['lower'], COLOR_THRESHOLDS[color_name]['upper'])
    else:
        raise ValueError(f"Color '{color_name}' not supported. Try: {list(COLOR_THRESHOLDS.keys())}")
        
    return mask

def morphological_clean(mask):
    """Clean up noise in the mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Close gaps inside the object
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Open to remove small noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def find_largest_contour_centroid(mask):
    """Find the centroid (u, v) of the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Get the largest contour by area
    c = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(c) < 100: # Ignore tiny noise
        return None
        
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)
