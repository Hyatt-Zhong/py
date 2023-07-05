import win32gui
import sys
import time as tm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *

handle = win32gui.FindWindow(None, "任务管理器")
app = QApplication(sys.argv)
screen = QApplication.primaryScreen()
QApplication.primaryScreen
for i in range(1):
    tm.sleep(0.25)
    # img = screen.grabWindow(handle).toImage()
    img = screen.grabWindow(0).toImage()
    img.save("screenshot.jpg")
