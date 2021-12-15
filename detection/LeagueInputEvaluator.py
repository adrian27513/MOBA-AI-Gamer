import argparse
import os
import sys
from pathlib import Path

import os
import math
from LeagueAI_helper import input_output
import tesserocr
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import numpy as np
import time

import cv2
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
IO = input_output(input_mode='videofile', video_filename="AttributeDetection.mp4")

frame, ret = IO.get_pixels()
api = tesserocr.PyTessBaseAPI()

minions = 0
kills = 0
deaths = 0
assists = 0

device = select_device('')
weights = 'LeagueAIWeights.pt'
imgsz = [1080, 1920]
original_size = imgsz
model = DetectMultiBackend(weights, device=device, dnn=False)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size
print(f"Original size: {original_size}")
print(f"YOLO input size: {imgsz}")

half = False
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
classes = None
agnostic_nms=False,  # class-agnostic NMS
max_det=1000

model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
dt, seen = [0.0, 0.0, 0.0], 0

while ret:
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame, ret = IO.get_pixels()
    if ret == True:
        detection_results = []
        attributes = [minions, kills, deaths, assists]
        results = [attributes, detection_results]

        #Object Detection
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img2 = img.resize([imgsz[1], imgsz[0]], Image.ANTIALIAS)

        img_raw = torch.from_numpy(np.asarray(img2)).to(device)
        img_raw = img_raw.half() if half else img_raw.float()  # uint8 to fp16/32
        img_raw /= 255  # 0 - 255 to 0.0 - 1.0
        img_raw = img_raw.unsqueeze_(0)
        img_raw = img_raw.permute(0, 3, 1, 2)

        pred = model(img_raw, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(original_size, det[:, :4], imgsz).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # Choose between xyxy and xywh as your desired format.
                    detection_results.append([names[int(cls)], float(conf), [*xyxy]])
        results[1] = detection_results

        # Attribute Detection
        # Minions
        Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
        crop_minions = frame[My1:My2, Mx1:Mx2]
        # KDA
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
                minions = int(output.strip())
            elif count == 1:
                kda = output.split('/')
                kills = int(kda[0])
                deaths = int(kda[1])
                assists = int(kda[2])

        attributes = [minions, kills, deaths, assists]
        results[0] = attributes

        print(results)
