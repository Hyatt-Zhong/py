import cv2
import numpy as np

def read_gray_image(pic):
    image = cv2.imread(pic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 读取图像
# image = read_gray_image('image.jpg')
image = cv2.imread("image.jpg",cv2.IMREAD_UNCHANGED)

# 生成随机噪点
# noise = np.zeros_like(image, np.uint8)
# cv2.randn(noise, 0, 0xff)  # 生成均值为0，标准差为50的随机数组

# # 将噪点加入图像
# noisy_image = cv2.add(image, noise)
# print(image[50])
# print(noisy_image[50])

count = np.zeros_like(image, np.int8)


for i in range(0,20):
    noise = np.zeros_like(image, np.uint8)
    cv2.randn(noise, 0, 0xff)  # 生成均值为0，标准差为50的随机数组

    noisy_image = cv2.add(image, noise)
    count[noisy_image>6] +=1
    count -= 1

output = np.zeros_like(image, np.uint8)
output[count>=0]=255

# # 显示结果
cv2.imshow('Noisy Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(image[0])
# print(noise[0])