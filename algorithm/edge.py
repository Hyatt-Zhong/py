import cv2

# 读取原始图像
img = cv2.imread('mario.jpg')

# 转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # 对图像进行高斯模糊以改善边缘检测
# img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# # 计算X轴和Y轴上的Sobel边缘
# sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# # 显示Sobel边缘检测结果
# cv2.imshow('Sobel X', sobelx)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)

# 对图像进行高斯模糊以改善边缘检测
edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)

# 显示Canny边缘检测结果
cv2.imshow('Canny Edge Detection', img_gray)
cv2.waitKey(0)