from adrian_LeagueAI_helper import input_output
import tesserocr
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import numpy as np
import time
import pyautogui
import pydirectinput
import random
import cv2
import torch

pyautogui.FAILSAFE = True

# Minions  Kills  Deaths  Assists
attributes = [0, 0, 0, 0]
api = tesserocr.PyTessBaseAPI()

model = torch.hub.load('C:\\Users\\Adrian\\PycharmProjects\\AdrianLeagueAIYoloV5\\yolov5', 'custom', path='LeagueAIWeights.pt', source='local')

model.conf = 0.6  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = [0,2,3,5,7,9,11,12,14,16,18]   # Remove dead classes
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# input = 'video'
input = 'desktop'
if input == 'video':
    IO = input_output(input_mode='videofile', video_filename="AttributeDetection.mp4")
elif input == 'desktop':
    IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080)

frame, ret = IO.get_pixels()
while ret:
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame, ret = IO.get_pixels()
    if ret == True:
        if input == 'video':
            #Minions
            Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
            crop_minions = frame[My1:My2, Mx1:Mx2]
            #KDA
            Kx1, Ky1, Kx2, Ky2 = 1663, 0, 1718, 25
            crop_kda = frame[Ky1:Ky2, Kx1:Kx2]
        elif input == 'desktop':
            Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
            crop_minions = frame.crop((Mx1, My1, Mx2, My2))
            Kx1, Ky1, Kx2, Ky2 = 1663, 0, 1718, 25
            crop_kda = frame.crop((Kx1, Ky1, Kx2, Ky2))

        info = [crop_minions, crop_kda]
        for count, i in enumerate(info):
            image = i
            if input == 'desktop':
                image = np.array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Perform text extraction
            attribute_img = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))

            with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
                api.SetImage(attribute_img)
                output = api.GetUTF8Text()

            if count == 0:
                try:
                    attributes[0] = int(output.strip())
                except:
                    attributes[0] = attributes[0]
            elif count == 1:
                try:
                    kda = output.split('/')
                    attributes[1] = int(kda[0])
                    attributes[2] = int(kda[1])
                    attributes[3] = int(kda[2])
                except:
                    attributes[1] = attributes[1]
                    attributes[2] = attributes[2]
                    attributes[3] = attributes[3]

        if input == 'video':
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(pil_img)  # custom inference size
        elif input == 'desktop':
            results = model(frame)

        object_results = results.pandas().xyxy[0]
        print(object_results)
        print(attributes)

        red_minion_idx = object_results.index[object_results['name']=='red_melee'].tolist()
        if not len(red_minion_idx) == 0:
            center = [(object_results['xmin'][red_minion_idx[0]] + object_results['xmax'][red_minion_idx[0]]) / 2,
                        (object_results['ymin'][red_minion_idx[0]] + object_results['ymax'][red_minion_idx[0]]) / 2]
            pyautogui.click(button='right', x=center[0], y=center[1])
            abilities = ['q', 'w','e','r']
            ability = random.randint(0,3)
            pydirectinput.press(abilities[ability])
        else:
            pyautogui.mouseUp(button='right')


