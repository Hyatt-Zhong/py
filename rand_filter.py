import cv2
import numpy as np

def read_gray_image(pic):
    image = cv2.imread(pic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 读取图像
image = read_gray_image('image.jpg')

# 生成随机噪点
noise = np.zeros_like(image, np.uint8)
cv2.randn(noise, 0, 0xff)  # 生成均值为0，标准差为50的随机数组

# 将噪点加入图像
noisy_image = cv2.add(image, noise)

# # 显示结果
# cv2.imshow('Noisy Image', noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(image[0])
# print(noise[0])