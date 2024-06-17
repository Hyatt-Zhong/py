

import cv2
import numpy as np
import threading
import queue
import time

# 创建一个线程安全的队列
image_queue = queue.Queue()

def display_window():
    # 创建一个固定窗口
    cv2.namedWindow('Fixed Window', cv2.WINDOW_NORMAL)

    while True:
        # 如果队列不为空，从队列中获取最新的图片
        if not image_queue.empty():
            image = image_queue.get()
            # 清空窗口并显示新图片
            cv2.imshow('Fixed Window', image)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def update_images():
    while True:
        # 模拟从某个地方获取图片
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        image_queue.put(image)
        time.sleep(1)  # 模拟处理时间

def main():

   # 子线程：从队列中读取图片并更新窗口内容
    update_thread = threading.Thread(target=update_images)
    update_thread.daemon = True  # 设置为守护线程，这样程序退出时线程会自动结束
    update_thread.start()

    display_window()

main()