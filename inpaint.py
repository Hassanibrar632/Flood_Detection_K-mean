import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_vertical_patches(image, patch_width, overlap):
    """
    Divide an image into overlapping vertical patches.
    
    Args:
        image (numpy array): The input image (read using cv2.imread or similar).
        patch_width (int): The width of each patch.
        overlap (int): The overlap (in pixels) between adjacent patches.
    
    Returns:
        list: A list of vertical patches (each patch is a numpy array).
    """
    patches = []
    _, img_width = image.shape[:2]
    
    # Step size based on overlap
    step_x = patch_width - overlap
    
    # Loop through the image horizontally and extract vertical patches
    for x in range(0, img_width - patch_width + 1, step_x):
        patch = image[:, x:x + patch_width]  # Full height, specific width
        patches.append(patch)
    
    # Handle the case where the last patch might be missed
    if (img_width - patch_width) % step_x != 0:
        patch = image[:, img_width - patch_width:img_width]
        patches.append(patch)
    
    return patches

def apply_canny_edges(image, lower_threshold, upper_threshold):
    """
    Apply Canny edge detection to an image.
    
    Args:
        image (numpy array): Input grayscale image.
        lower_threshold (int): Lower threshold for edge detection.
        upper_threshold (int): Upper threshold for edge detection.
    
    Returns:
        numpy array: The edges detected in the image.
    """
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

if __name__ == "__main__":
    # Load the image
    image_path = "test.jpg"  # Replace with your image file path
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
    
    # Parameters
    patch_width = 50    # Width of each vertical patch
    overlap = 25        # Overlap in pixels
    K = 3               # Number of centroids for k-means
    attempts = 50       # Iterations for the clustering algorithm
    lower_threshold = 100  # Lower threshold for Canny
    upper_threshold = 200  # Upper threshold for Canny

    # K-means criteria (stopping criteria)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Get vertical patches
    vertical_patches = get_vertical_patches(img, patch_width, overlap)
    
    # Process each patch with k-means clustering and Canny edges
    for i, patch in enumerate(vertical_patches):
        # Convert the patch to grayscale for Canny
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        edges = apply_canny_edges(patch_gray, lower_threshold, upper_threshold)
        
        # Vectorize the patch for k-means
        vectorized = patch.reshape((-1, 3)).astype(np.float32)
        
        # Apply k-means clustering
        _, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        
        # Reshape the results back to the patch size
        res = center[label.flatten()]
        result_image = res.reshape((patch.shape))
        
        # Plot the original patch, edge-detected patch, and k-means result
        figure_size = 15
        plt.figure(figsize=(figure_size, figure_size))
        
        plt.subplot(1, 2, 1)
        plt.imshow(patch)
        plt.title(f'Original Patch {i+1}'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_image.astype(np.uint8))
        plt.title(f'Segmentations of Patch {i+1}'), plt.xticks([]), plt.yticks([])
        
        plt.show()
