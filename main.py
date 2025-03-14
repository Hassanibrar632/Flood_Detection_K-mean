# importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Generate Vertical Patches of the image so that we can use them to get water levels
def get_vertical_patches(image, patch_width, overlap):
    patches = []
    _, img_width = image.shape[:2]
    step_x = patch_width - overlap
    for x in range(0, img_width - patch_width + 1, step_x):
        patch = image[:, x:x + patch_width]
        patches.append(patch)
    if (img_width - patch_width) % step_x != 0:
        patch = image[:, img_width - patch_width:img_width]
        patches.append(patch)
    return patches

# Calculate the water levels of the patches depending on the clusters and the water pixels range
def calculate_water_height_from_bottom(strip, water_intensity_range=(70, 255)):
    lower, upper = water_intensity_range
    water_mask = cv2.inRange(strip, lower, upper)
    water_height = 0
    
    for i in range(len(water_mask) - 1, -1, -1):
        if water_mask[i] > 0:
            water_height += 1
        else:
            break
    
    return water_height

if __name__ == "__main__":
    image_path = r"EXTRACTED_FRAMES\frame_397.jpg"
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters
    patch_width = 50
    overlap = 25
    K = 8
    attempts = 50
    water_intensity_range = (50, 255)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Get vertical patches
    vertical_patches = get_vertical_patches(img, patch_width, overlap)
    water_heights = []

    for i, patch in enumerate(vertical_patches):
        # Vectorize the patch for k-means
        vectorized = patch.reshape((-1, 1)).astype(np.float32)
        
        # Apply k-means clustering
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        res = center[label.flatten()]
        result_image = res.reshape((patch.shape))
        
        # Get a vector of the mean cluster values of all rows
        strip = np.array(result_image).mean(axis=1)
        
        # Calculate water height from bottom to top depending on the values of the strip and the water range
        water_height = calculate_water_height_from_bottom(strip, water_intensity_range)
        water_heights.append(water_height)

        # Plot all visualizations in a single row
        figure_size = 20
        plt.figure(figsize=(figure_size, 5))

        # Orignal Patch
        plt.subplot(1, 4, 1)
        plt.imshow(patch, cmap="gray")
        plt.title(f'Original Patch {i+1}'), plt.xticks([]), plt.yticks([])

        # Segmentated Image
        plt.subplot(1, 4, 2)
        plt.imshow(result_image, cmap="gray")
        plt.title(f'Segmentation {i+1}'), plt.xticks([]), plt.yticks([])

        # Strip plot of the mean value range and distribution
        plt.subplot(1, 4, 3)
        plt.imshow(strip.reshape(-1, 1), cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('Row-wise Sum Color Map')
        plt.axis('off')

        # Add bar chart based on the water level detected
        plt.subplot(1, 4, 4)
        patch_height = patch.shape[0]
        plt.bar([i+1], [water_height], color='blue', alpha=0.7)
        plt.xlabel('Patch')
        plt.ylabel('Height (px)')
        plt.title('Water Height from Bottom')
        plt.ylim(0, patch_height)
        plt.xticks([i+1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

        exit()
