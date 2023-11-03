

# import torch

# # 创建一个张量
# x = torch.tensor([1, 6, 3, 8, 2, 4])
# y = torch.tensor([1, 6, 3, 8, 2, 4])

# # 使用逻辑运算符和索引操作来对元素赋值
# x[y > 5] = 0

# print(x)


import cv2
import numpy as np
from skimage.feature import graycomatrix,graycoprops

# 读取图像并转换为灰度图像
image = cv2.imread('desktop.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算灰度共生矩阵
glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=False)
aa=glcm[:,:,0,0].astype(np.uint8)
# print(gray[0],glcm[:,0,0,0])
print(gray.dtype, gray.shape,aa.dtype,aa.shape)

contrast = graycoprops(glcm, 'all')
energy = graycoprops(glcm, 'energy')
# entropy = graycoprops(glcm, 'entropy')
print("contrast","energy","entropy")
print(contrast,energy,"entropy")

cv2.imshow('Result', aa)
cv2.waitKey(0)
cv2.destroyAllWindows()