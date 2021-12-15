#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from mss import mss
from PIL import Image
import cv2
import numpy as np

class detection:
    def __init__(self, object_class, x_min, y_min, x_max, y_max):
        self.object_class = object_class
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.w = abs(x_max - x_min)
        self.h = abs(y_max - y_min)
        self.x = x_min + int(self.w/2)
        self.y = y_min + int(self.h/2)
    def toString(self):
        print("class: {}, min: ({}|{}), max: ({}|{}), width: {}, height: {}, center: ({}|{})".format(self.object_class, self.x_min, self.y_min, self.x_max, self.y_max, self.w, self.h, self.x, self.y))

class input_output:
    def __init__(self, input_mode, SCREEN_WIDTH=None, SCREEN_HEIGHT=None, video_filename=None):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.input_mode = input_mode
        self.video_filename = video_filename
        if input_mode == 'webcam':
            self.capture_device = cv2.VideoCapture(0)
            assert(self.capture_device.isOpened()), 'Error could not open capture device for Webcam -1'
        elif input_mode == 'videofile':
            assert(self.video_filename is not None), "Error please enter a valid video file name"
            self.capture_device = cv2.VideoCapture(self.video_filename)
            assert self.capture_device.isOpened(), 'Error could not open capture device for Videofile: {}'.format(self.video_filename)
        elif input_mode == 'desktop':
            assert(SCREEN_HEIGHT is not None and SCREEN_HEIGHT is not None), "Error please set SCREEN_WIDTH and SCREEN_HEIGHT"
            self.capture_device = mss()
            self.mon = {'top': 0, 'left': 0, 'width' : self.SCREEN_WIDTH, 'height' : self.SCREEN_HEIGHT}
        else:
            raise Exception('Unknown input mode!')

    def get_pixels(self, output_size=None):
        if self.input_mode == 'webcam':
            ret, frame = self.capture_device.read()
            #assert(ret == True), 'Error: could not retrieve frame'
            return frame, ret
        if self.input_mode == 'videofile':
            ret, frame = self.capture_device.read()
            #assert(ret == True), 'Error: could not retrieve frame'
            return frame, ret
        elif self.input_mode == 'desktop':
            frame = self.capture_device.grab(self.mon)
            screen = Image.frombytes('RGB', frame.size, frame.bgra, "raw", "BGRX")
            # Swap R and B channel
            R, G, B = screen.split()
            screen = Image.merge("RGB", [B, G, R])
            screen = np.array(screen)
            if output_size == None:
                output_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            screen = cv2.resize(screen, output_size)
            return screen
        else:
            raise Exception('Unknown input mode!')

import torch
import torch.nn as nn
from torch.autograd import Variable

class LeagueAIFramework():
    def __init__(self, config_file, weights, names_file, classes_number, resolution, threshold, cuda=True, nms_confidence=0.4, draw_boxes=True):
        self.config_file = config_file
        self.weights = weights
        self.names_file = names_file
        self.names = self.load_classes(self.names_file)
        self.classes = classes_number
        self.resolution = resolution
        assert (self.resolution > 0), "Error: Resolution out of bounds"
        self.threshold = threshold
        assert (self.threshold >= 0 or self.threshold > 1.0), "Error, threshold out of bounds!"
        # Create a new detector object with the given config file
        self.detector = Detector(self.config_file)
        print("Created new detector with config: ", self.config_file)
        # Load the weights and set up the network
        self.detector.load_weights(self.weights)
        self.detector.net_info["height"] = self.resolution
        # Make sure the resolution is a multiple of 32
        self.input_dimension = int(self.detector.net_info["height"])
        assert (self.input_dimension % 32 == 0 and self.input_dimension > 32), "Error the resolution is not a multiple of 32!"
        self.cuda = cuda
        self.nms_conf = nms_confidence
        self.draw_boxes = draw_boxes
        print("Network ", weights, " loaded succesfully!")
        if self.cuda:
            if torch.cuda.is_available():
                self.detector.cuda()
                print("Model running on GPU")
            else:
                self.cuda = False
                print("Cuda not available, running on CPU")
        else:
            print("Model running on CPU")
