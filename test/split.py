import cv2
import numpy as np

# 读取图像并将其转换为HSV颜色空间
image = cv2.imread("desktop.jpg")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取图像中每个像素的H通道值
h_channel = hsv_image[:, :, 0]

# 创建一个与原始图像大小相同的零矩阵，用于存储边缘结果
edges = np.zeros_like(h_channel)



# 遍历图像的每个像素，并检查其与相邻像素的H通道值差异
for i in range(1, hsv_image.shape[0]-1):
    for j in range(1, hsv_image.shape[1]-1):
        h_diff = np.abs(h_channel[i, j] - h_channel[i:i+2, j:j+2])
        if np.any(h_diff > 55):
            edges[i, j] = 255

# 显示边缘图像
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()