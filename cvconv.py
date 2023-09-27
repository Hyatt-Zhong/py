import cv2

# 读取图像
image = cv2.imread('desktop.jpg', 0)

# 使用Otsu算法进行阈值分割
_, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示结果
cv2.imshow('Threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()