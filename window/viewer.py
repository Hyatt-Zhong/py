import cv2
import threading
import queue
from mttkinter import mtTkinter as tk
from PIL import Image, ImageTk

width = 900
height = 500

strsize = str(width) + "x" + str(height)


def get_img(que):
    if not que.empty():
        img = que.get()
        # 将OpenCV图像转换为PIL图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # 将PIL图像转换为Tkinter图像
        img = ImageTk.PhotoImage(img)
        return img
    else:
        return None


def display_image(image_label, que, lck):
    lck.acquire()
    img = get_img(que)
    lck.release()
    if img != None:
        # 更新标签中的图像
        image_label.configure(image=img)
        image_label.image = img

    # 每隔一段时间调用自身以实现循环
    image_label.after(100, display_image, image_label, que, lck)


def popup_menu(event):
    window = event.widget
    menu = tk.Menu(window, tearoff=0)
    menu.add_checkbutton(label="菜单项1")
    menu.add_checkbutton(label="菜单项2")
    menu.add_separator()
    menu.add_command(label="退出", command=window.quit)
    menu.post(event.x_root, event.y_root)


def viewfunc(que, lck):
    window = tk.Tk()
    window.title("OpenCV 图片展示")
    window.configure(bg="black")
    window.geometry(strsize)
    window.resizable(False, False)

    image_label = tk.Label(window)
    image_label.pack()
    image_label.configure(bg="black")

    display_image(image_label, que, lck)

    window.bind("<Button-3>", popup_menu)

    window.mainloop()


def put_img(que, img, lck):
    lck.acquire()
    if que.qsize() >= 20:
        tmp = que.get()
        que.put(img)
    else:
        que.put(img)
    lck.release()


def resize(img):
    h = img.shape[0]
    w = img.shape[1]
    # print(h/w,height/width)
    rx=1

    if h / w > height / width:
        rx=height/h
    else:
        rx=width/w
    
    img = cv2.resize(img, None, fx=rx,fy=rx)

    return img


# que = queue.Queue()
# lock = threading.Lock()
# image = cv2.imread("screenshot.jpg", cv2.IMREAD_UNCHANGED)
# put_img(que, image, lock)

# thread = threading.Thread(target=viewfunc, args=(que, lock))
# thread.start()
# thread.join()
