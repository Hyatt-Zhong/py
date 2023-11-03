import cv2
import numpy as np


def max_pooling(img, G=10):
 
    # Max Pooling
 
    out = img.copy()
 
    H, W, C = img.shape
 
    Nh = int(H / G)
 
    Nw = int(W / G)
 
    for y in range(Nh):
 
        for x in range(Nw):
 
            for c in range(C):
 
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])
 
    return out

def read_gray_image(pic):
    image = cv2.imread(pic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
# OpenCV按能量分割图块
# img = cv2.imread('DT.jpg')
img = read_gray_image('DT.jpg')

# 创建锐化核
kernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])

# kernel = np.array([[1/9,1/9,1/9],
#                    [1/9,1/9,1/9],
#                    [1/9,1/9,1/9]])

sharpened_img = cv2.filter2D(img, -1, kernel)


# 进行最大池化
# sharpened_img = max_pooling(img)

# kernel = np.array([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])

# 应用锐化核
# for i in range(0,100):
#     img=sharpened_img
#     sharpened_img = cv2.filter2D(img, -1, kernel)

# gray = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Result', img)
cv2.waitKey(0)

# _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#     cv2.rectangle(thresh, (x, y), (x+w, y+h), (122), 1)

# cv2.imshow('Result', thresh)
# cv2.waitKey(0)
cv2.destroyAllWindows()







# 在hsv空间上查找图像边缘

# import cv2
# import numpy as np

# # 读取图像并将其转换为HSV颜色空间
# image = cv2.imread("desktop.jpg")
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # 提取图像中每个像素的H通道值
# h_channel = hsv_image[:, :, 0]

# # 创建一个与原始图像大小相同的零矩阵，用于存储边缘结果
# edges = np.zeros_like(h_channel)



# # 遍历图像的每个像素，并检查其与相邻像素的H通道值差异
# for i in range(1, hsv_image.shape[0]-1):
#     for j in range(1, hsv_image.shape[1]-1):
#         h_diff = np.abs(h_channel[i, j] - h_channel[i:i+2, j:j+2])
#         if np.any(h_diff > 55):
#             edges[i, j] = 255

# # 显示边缘图像
# cv2.imshow("Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()