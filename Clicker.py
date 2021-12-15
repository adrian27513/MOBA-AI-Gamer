import pyautogui
import pydirectinput
import time
import random
time.sleep(5)
pyautogui.FAILSAFE = True

print(pyautogui.position())
#1555, 230
#1650, 250

while True:
    randomTime = random.random() * 10
    randomLocationX = random.randint(1555, 1650)
    randomLocationY = random.randint(230, 250)
    pyautogui.click(randomLocationX, randomLocationY)
    time.sleep(randomTime)
