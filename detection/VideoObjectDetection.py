import argparse
import os
import sys
from pathlib import Path

import os
import math
from a_LeagueAI_helper import input_output
import tesserocr
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import numpy as np
import time

import cv2
import torch

model = torch.hub.load('C:\\Users\\Adrian\\PycharmProjects\\AdrianLeagueAIYoloV5\\yolov5', 'custom', path='LeagueAIWeights.pt', source='local')

model.conf = 0.7  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = [0,2,3,5,7,9,11,12,14,16,18]   # Remove dead classes
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

input = 'video'
# input = 'desktop'
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
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(pil_img)  # custom inference size
        elif input == 'desktop':
            results = model(frame)

        object_results = results.pandas().xyxy[0]
        print(object_results)
        # print(object_results.get('xmin')[0])