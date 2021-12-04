import pyautogui
import time
import random
time.sleep(5)
pyautogui.FAILSAFE = True

while True:
    x = random.randint(740, 1525)
    y = random.randint(565, 795)
    timer = random.random() * 10
    pyautogui.click(x,y)
    time.sleep(timer)