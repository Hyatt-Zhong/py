import cv2
import numpy as np

# 读取图片
image = cv2.imread('desktop.jpg')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊减少图像噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 应用Canny边缘检测
edges = cv2.Canny(blurred, 130, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓并绘制边界框
for contour in contours:
    # 计算每个轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 在原图上绘制矩形
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 显示图像
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 高斯模糊：cv2.GaussianBlur 用于对图片进行模糊处理。这可以帮助减少图像中的噪声，从而在随后的边缘检测中减少误检。
# Canny边缘检测：cv2.Canny 是用于边缘检测的函数，其中的两个阈值（这里是50和150）用于确定何种梯度的强度被认为是边缘。这两个值可以根据你的图像进行调整。