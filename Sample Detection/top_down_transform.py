import cv2
import numpy as np

# Load the image
image = cv2.imread('upscaled2.png')
calib_width = 1024
calib_height = 768

# Resize the image to match the calibration resolution
resized_image = cv2.resize(image, (calib_width, calib_height))

tl = (250, 320)
bl = (0, calib_height)
tr = (calib_width - tl[0], tl[1])
br = (calib_width - bl[0], bl[1])

cv2.circle(resized_image, tl, 5, (0,0,255), -1)
cv2.circle(resized_image, bl, 5, (0,0,255), -1)
cv2.circle(resized_image, tr, 5, (0,0,255), -1)
cv2.circle(resized_image, br, 5, (0,0,255), -1)

pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0,0], [0,calib_height], [calib_width,0], [calib_width,calib_height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
transformed_frame = cv2.warpPerspective(resized_image, matrix, (calib_width,calib_height))

# Show the result
cv2.imshow('Undistorted', resized_image)
cv2.imshow('Top-Down View', transformed_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
