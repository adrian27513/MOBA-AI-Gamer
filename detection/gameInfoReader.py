import os
import math
from a_LeagueAI_helper import input_output
import tesserocr
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import numpy as np
import cv2
import time

# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
IO = input_output(input_mode='videofile', video_filename="AttributeDetection.mp4")

minions = 0
kills = 0
deaths = 0
assists = 0

frame, ret = IO.get_pixels()

api = tesserocr.PyTessBaseAPI()

while ret:
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame, ret = IO.get_pixels()
    if ret == True:
        #Minions
        Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
        crop_minions = frame[My1:My2, Mx1:Mx2]
        #KDA
        Kx1, Ky1, Kx2, Ky2 = 1663, 0, 1718, 25
        crop_kda = frame[Ky1:Ky2, Kx1:Kx2]

        info = [crop_minions, crop_kda]

        for count, i in enumerate(info):
            image = i
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Perform text extraction
            pil_img = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))

            with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
                api.SetImage(pil_img)
                output = api.GetUTF8Text()

            if count == 0:
                minions = output.strip()
            elif count == 1:
                kda = output.split('/')
                kills = kda[0]
                deaths = kda[1]
                assists = kda[2]

            print("Minions: " + minions)
            print("Kills: " + str(kills))
            print("Deaths: " + str(deaths))
            print("Assists: " + str(assists))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break