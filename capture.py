import win32gui
import sys
import time as tm
import window.viewer as vr
import numpy as np
import queue
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *

# handle = win32gui.FindWindow(None, "任务管理器")
app = QApplication(sys.argv)
screen = QApplication.primaryScreen()
# for i in range(1):
#     tm.sleep(0.25)
#     # img = screen.grabWindow(handle).toImage()
#     img = screen.grabWindow(0).toImage()
#     img.save("screenshot.jpg")

que = queue.Queue()
lock = threading.Lock()

thread = threading.Thread(target=vr.viewfunc, args=(que, lock))
thread.daemon = True
thread.start()

def convertQImageToMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4) 
    return arr

# img = screen.grabWindow(0).toImage()
# mat=convertQImageToMat(img)

# vr.cv_show(mat)

while True:
    tm.sleep(0.25)
    # img = screen.grabWindow(handle).toImage()
    img = screen.grabWindow(0).toImage()
    mat=convertQImageToMat(img)

    mat=vr.resize(mat)
    vr.put_img(que,mat,lock)

