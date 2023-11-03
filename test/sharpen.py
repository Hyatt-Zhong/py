import cv2
import numpy as np

# 加载图像
# img = cv2.imread('screenshot.jpg')
img = cv2.imread('text.jpg')
# img = cv2.imread('image.jpg')

# 创建锐化核
kernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])

# kernel = np.array([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])

# 应用锐化核
sharpened_img = cv2.filter2D(img, -1, kernel)

diff=sharpened_img-img

# 显示原始图像和锐化后的图像
# cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image', sharpened_img)
# cv2.imshow('diff', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()