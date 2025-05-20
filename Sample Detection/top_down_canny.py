import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "4.png"
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

# Apply Canny edge detection
edges = cv2.Canny(gray, 30, 100)

# Keep edges only within the yellow mask
mask_blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
edges[mask_blurred == 0] = 0

# Morphological operations to connect edges
kernel = np.ones((5, 5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
edges = cv2.dilate(edges, kernel, iterations=2)

# Find contours from edge map
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and approximate to quadrilaterals
top_faces = []
if hierarchy is not None:
    hierarchy = hierarchy[0]  # Remove extra nesting

    for i, cnt in enumerate(contours):
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)

        if len(approx) == 4: # and is_rectangle(approx, 50)
            # Compute vertical center of the shape
            ys = [pt[0][1] for pt in approx]
            y_center = np.mean(ys)
            height = image.shape[0]

            if area > 4000:
                cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)
                top_faces.append(approx)

# Show results
bgr_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Draw all found contours on a copy of the image for visualization
contour_display = image.copy()
cv2.drawContours(contour_display, contours, -1, (0, 0, 255), 2)  # red contours

# Convert BGR to RGB for display
contour_display_rgb = cv2.cvtColor(contour_display, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(24, 8))

ax[1].imshow(edges_display)
ax[1].set_title("Canny Edge Detection")
ax[1].axis("off")

ax[0].imshow(bgr_image)
ax[0].set_title("Detected Top Faces of Yellow Boxes")
ax[0].axis("off")

plt.tight_layout()
plt.show()