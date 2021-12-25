import pyautogui
import pydirectinput
import time
import os
from os import listdir

pyautogui.FAILSAFE = True

def reset_camera():
    pyautogui.click(1665, 1000)
    time.sleep(1)
    pyautogui.doubleClick(1690, 815)
    time.sleep(1)
    pyautogui.doubleClick(1870, 920)
    pyautogui.keyDown('alt')
    time.sleep(1)
    pydirectinput.press('up', presses=8)
    pyautogui.keyUp('alt')
    pyautogui.scroll(-200)


def reset_properties():
    pyautogui.click(1720, 645)
    time.sleep(3)
    pyautogui.click(290, 215)
    time.sleep(1)
    pyautogui.click(755, 565)
    time.sleep(1)
    pyautogui.click(225, 64)
    time.sleep(1)
    pyautogui.doubleClick(265, 160)


def chooseAnimation():
    animationLocation = pyautogui.locateOnScreen(
        'MOBAAIGamer\\generate_dataset\AnimationLogo.PNG')
    animationCenter = pyautogui.center(animationLocation)
    pyautogui.click(animationCenter)
    time.sleep(1)
    animation = list(pyautogui.locateAllOnScreen(
        'MOBAAIGamer\\generate_dataset\AnimationModel.PNG'))
    pyautogui.click(1870, 920)
    for x in animation:
        pyautogui.click(animationCenter)
        time.sleep(1)
        animationModelCenter = pyautogui.center(x)
        pyautogui.doubleClick(animationModelCenter)
        pyautogui.moveTo(1870, 920)
        pyautogui.hotkey('win', 'alt', 'r')
        time.sleep(20)
        pyautogui.hotkey('win', 'alt', 'r')
        time.sleep(7)


def openFile(fileName, first):
    os.startfile(fileName)
    time.sleep(30)
    if first:
        reset_properties()
    reset_camera()
    chooseAnimation()

first = True
modelPath = "E:\\Adrian\\LeagueAI\\RecordedData"
modelDirectory = sorted(listdir(modelPath))
print(modelDirectory)
for innerModels in modelDirectory:
    innerModel = modelPath + "\\" + innerModels
    innerModelDirectory = sorted(listdir(innerModel))
    print(innerModelDirectory)
    for models in innerModelDirectory:
        model = innerModel + "\\" + models + "\\" + models + ".gltf"
        if (first):
            print(model)
            openFile(model, first)
            first = False
        else:
            print(model)
            openFile(model, first)

