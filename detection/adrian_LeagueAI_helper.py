from mss import mss
from PIL import Image, ImageGrab
import cv2
import numpy as np

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
            screen = ImageGrab.grab()
            return screen, True
        else:
            raise Exception('Unknown input mode!')