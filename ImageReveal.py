import numpy as np 
import matplotlib.pyplot as plt
import cv2
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



# Phase 3: Image Processing
# Convert to uint8 for proper color conversion
image_uint8 = (image_data * 255).astype(np.uint8)
rgb_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

# Add features
rgb_image[5,5]=[255,0,0]
rgb_image[5,14]=[255,0,0]
rgb_image[15,6]=[0,0,0]
rgb_image[15,12]=[0,0,0]
rgb_image[13,6]=[242,243,255]
rgb_image[13,12]=[242,243,255]


# Add border
border_size = 10
border_color = (0, 0, 255)  # Blue in BGR format
bordered_image = cv2.copyMakeBorder(rgb_image, border_size, border_size, border_size, border_size,
                                   cv2.BORDER_CONSTANT, value=border_color)

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(bordered_image)
plt.title("Processed Image")
plt.axis('off')
plt.show()

# Save the image
cv2.imwrite('processed_image.png', cv2.cvtColor(bordered_image, cv2.COLOR_RGB2BGR))
print("Image saved as 'processed_image.png'")