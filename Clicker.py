import pyautogui
import pydirectinput
import time
import random
time.sleep(5)
pyautogui.FAILSAFE = True

letters = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
           'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'backspace')
while True:
    x = random.randint(740, 1525)
    y = random.randint(565, 795)
    timer = random.random() * 5
    key = letters[random.randint(0,len(letters) - 1)]
    choice = random.random()
    if choice < .25:
        pydirectinput.press(key)
    else:
        pyautogui.click(x,y)
    time.sleep(timer)