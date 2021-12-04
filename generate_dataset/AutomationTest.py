import pyautogui
import pydirectinput
import time
import os

pyautogui.FAILSAFE = True

def reset_camera():
    pyautogui.doubleClick(1870, 920)
    pyautogui.keyDown('alt')
    time.sleep(1)
    pydirectinput.press('up', presses=8)
    pyautogui.keyUp('alt')
    pyautogui.scroll(50)

time.sleep(5)
reset_camera()


