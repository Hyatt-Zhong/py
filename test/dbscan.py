import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('desktop.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用SIFT或其他特征检测器提取关键点
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)

# 将关键点转换为NumPy数组以便使用DBSCAN
points = np.array([kp.pt for kp in keypoints]).astype(np.float32)

# 应用DBSCAN聚类算法
dbscan = DBSCAN(eps=30, min_samples=5)
labels = dbscan.fit_predict(points)

# 为每个聚类绘制边界框
for label in np.unique(labels):
    if label == -1:
        # -1 表示噪声点
        continue

    # 找出同一聚类的所有点
    class_member_mask = (labels == label)
    xy = points[class_member_mask]

    # 绘制边界框
    rect = cv2.minAreaRect(xy)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()