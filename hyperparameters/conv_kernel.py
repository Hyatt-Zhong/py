import numpy as np

# Sobel卷积核：用于边缘检测，包括水平和垂直方向的Sobel卷积核。

# Laplacian卷积核：用于边缘检测和图像锐化。

# 高斯卷积核：用于图像平滑和降噪。

# Prewitt卷积核：用于边缘检测，包括水平和垂直方向的Prewitt卷积核。

# Roberts卷积核：用于边缘检测，包括水平和垂直方向的Roberts卷积核。

# Scharr卷积核：用于边缘检测，包括水平和垂直方向的Scharr卷积核。

# LoG（Laplacian of Gaussian）卷积核：用于边缘检测和图像锐化，结合了高斯平滑和Laplacian边缘检测。

# 梯度卷积核：用于计算图像的梯度，包括水平和垂直方向的梯度卷积核。

# Sobel卷积核
sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

sobel_vertical = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# Laplacian卷积核
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

# 高斯卷积核
gaussian = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])

# Prewitt卷积核
prewitt_horizontal = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])

prewitt_vertical = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

# Roberts卷积核
roberts_horizontal = np.array([[0, 1],
                               [-1, 0]])

roberts_vertical = np.array([[1, 0],
                             [0, -1]])

# Scharr卷积核
scharr_horizontal = np.array([[-3, 0, 3],
                              [-10, 0, 10],
                              [-3, 0, 3]])

scharr_vertical = np.array([[-3, -10, -3],
                            [0, 0, 0],
                            [3, 10, 3]])

# LoG（Laplacian of Gaussian）卷积核
log = np.array([[0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]])

# 梯度卷积核
gradient_horizontal = np.array([[-1, 0, 1]])

gradient_vertical = np.array([[-1],
                              [0],
                              [1]])