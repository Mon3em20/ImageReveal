import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Phase 1: Load and Reveal the Image
# -----------------------------------
# Load the CSV file as a NumPy array
image_data = np.loadtxt("secret_image .csv", delimiter=",")
plt.figure(figsize=(12, 8))

# Display grayscale image
plt.subplot(2, 2, 1)
plt.imshow(image_data, cmap="gray")
plt.title("Grayscale Image")
plt.colorbar()

# Display colormap: hot
plt.subplot(2, 2, 2)
plt.imshow(image_data, cmap="hot")
plt.title("Hot Colormap")
plt.colorbar()

# Display colormap: cool
plt.subplot(2, 2, 3)
plt.imshow(image_data, cmap="cool")
plt.title("Cool Colormap")
plt.colorbar()

# Display colormap: viridis
plt.subplot(2, 2, 4)
plt.imshow(image_data, cmap="viridis")
plt.title("Viridis Colormap")
plt.colorbar()

# Save visualizations
plt.imsave('grayscale_image.png', image_data, cmap='gray')
plt.imsave('hot_colormap.png', image_data, cmap='hot')
plt.imsave('cool_colormap.png', image_data, cmap='cool')
plt.imsave('viridis_colormap.png', image_data, cmap='viridis')



# Phase 2: Pattern Detection and Analysis
# ----------------------------------------
# Count black pixels (value = 0)

black_pixels_count = np.sum(image_data == 0)
print("Number of black pixels1:", black_pixels_count)

# Extract coordinates of black pixels

black_pixel_coords = np.column_stack(np.where(image_data == 0))
print("Coordinates of black pixels:")
print(black_pixel_coords)

# Bounding box for black pixels
min_y, min_x = np.min(black_pixel_coords, axis=0)
max_y, max_x = np.max(black_pixel_coords, axis=0)

print(f"Bounding Box: (x: {min_x} to {max_x}, y: {min_y} to {max_y})")

# Analyze pattern for symmetry or recognizable features
# (e.g., looking for eyes, mouth)
rows, cols = image_data.shape
left_side = image_data[:, :cols // 2]
right_side = np.flip(image_data[:, cols // 2:], axis=0)
is_symmetric = np.array_equal(left_side, right_side)
if is_symmetric:
    print("The pattern shows symmetry, possibly a face.")
else:
    print("No clear symmetry detected.")