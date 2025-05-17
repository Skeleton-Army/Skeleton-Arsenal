import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "image.png"
image = cv2.imread(image_path)
original = image.copy()

# Convert to HSV and extract yellow regions
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([10, 70, 70])
upper_yellow = np.array([35, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Sobel edge detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
sobel_edges = np.uint8(np.clip(sobel_mag, 0, 255))

# Keep edges only within the yellow mask
blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
sobel_edges[blurred == 0] = 0

# Apply threshold to binarize edge map
_, binary_edges = cv2.threshold(sobel_edges, 35, 255, cv2.THRESH_BINARY)

# Morphological operations to connect edges
kernel = np.ones((5, 5), np.uint8)
binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
binary_edges = cv2.dilate(binary_edges, kernel, iterations=2)

# Find contours from Sobel edge map
contours, hierarchy = cv2.findContours(binary_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and approximate to quadrilaterals
top_faces = []
if hierarchy is not None:
    hierarchy = hierarchy[0]  # Remove extra nesting

    for i, cnt in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)

        if len(approx) == 4: # and is_rectangle(approx, 50)
            # Compute vertical center of the shape
            ys = [pt[0][1] for pt in approx]
            y_center = np.mean(ys)
            height = image.shape[0]

            # Adjust area thresholds based on y position
            scale = y_center / height  # 0.0 (top) to 1.0 (bottom)
            min_area = 1000 + scale * 3000
            max_area = 15000 + scale * 5000

            if min_area < area < max_area:
                cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)
                top_faces.append(approx)

# Show results
bgr_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
edges_display = cv2.cvtColor(binary_edges, cv2.COLOR_GRAY2RGB)

# Draw all found contours on a copy of the image for visualization
contour_display = image.copy()
cv2.drawContours(contour_display, contours, -1, (0, 0, 255), 2)  # red contours

# Convert BGR to RGB for display
contour_display_rgb = cv2.cvtColor(contour_display, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(24, 8))
# ax[0].imshow(mask_display)
# ax[0].set_title("Yellow Box Mask")
# ax[0].axis("off")

ax[1].imshow(edges_display)
ax[1].set_title("Sobel Edge Detection")
ax[1].axis("off")

# ax[2].imshow(contour_display_rgb)
# ax[2].set_title("Contours")
# ax[2].axis("off")

ax[0].imshow(bgr_image)
ax[0].set_title("Detected Top Faces of Yellow Boxes")
ax[0].axis("off")

plt.tight_layout()
plt.show()