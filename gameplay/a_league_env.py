import gym
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image, ImageGrab
from gym import spaces
import numpy as np
import torch
import pyautogui
import pydirectinput
import tesserocr
import cv2


class LeagueEnv (gym.Env):
    def __init__(self):
        super(LeagueEnv, self).__init__()

        # Minions  Kills  Deaths  Assists
        self.attributes = [0, 0, 0, 0]

        api = tesserocr.PyTessBaseAPI()

        self.model = torch.hub.load('C:\\Users\\Adrian\\PycharmProjects\\MOBAAIGamer\\yolov5', 'custom',
                                    path='C:\\Users\\Adrian\\PycharmProjects\\MOBAAIGamer\\LeagueAIWeights.pt',
                                    source='local')

        self.model.conf = 0.6  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.classes = [0, 2, 3, 5, 7, 9, 11, 12, 14, 16, 18]  # Remove dead classes
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 50  # maximum number of detections per image

        # (Q, W, E, R, Right-Click), Move Mouse
        self.action_space = spaces.Tuple((spaces.Discrete(6),spaces.Box(low=0, high=1, shape=[2,1],dtype=np.float32)))

        # Object detection output
        self.observation_space = spaces.Box(low=0, high=4000, shape=[100,6], dtype=np.float32)

    def step(self, action):
        if self.action_space.contains(action):
            buttonPress = action[0]
            moveLocX = int(action[1][0] * 1920)
            moveLocY = int(action[1][1] * 1080)

            #Move Mouse
            pyautogui.moveTo(x=moveLocX, y=moveLocY)

            #Press Buttons
            if buttonPress == 0:
                # Press Q
                pydirectinput.press('q')
                # print('q')
            elif buttonPress == 1:
                # Press W
                pydirectinput.press('w')
                # print('w')
            elif buttonPress == 2:
                # Press E
                pydirectinput.press('e')
                # print('e')
            elif buttonPress == 3:
                # Press R
                pydirectinput.press('r')
                # print('r')
            elif buttonPress == 4:
                # Right-Click
                pyautogui.click(button='right')
                # print('click')
            elif buttonPress == 5:
                print("do nothing")

            obs, att = self.getObservation()
            reward, done = self.reward(obs, att)

            newObs = torch.zeros([100, 6])

            newObs[:obs[0], :6] = obs.xyxy[0]
            return newObs, reward, done

    def reset(self):
        # Reset
        pydirectinput.keyDown('ctrl')
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('p')
        pydirectinput.keyUp('ctrl')
        pydirectinput.keyUp('shift')
        pydirectinput.keyUp('p')
        pydirectinput.keyDown('shift')
        pydirectinput.press('y', presses=5)
        pydirectinput.press('c')
        pydirectinput.keyUp('shift')
        pydirectinput.keyDown('ctrl')
        pydirectinput.press('q', presses=3)
        pydirectinput.press('w')
        pydirectinput.press('e')
        pydirectinput.press('r')
        pydirectinput.keyUp('ctrl')

        obs, _ = self.getObservation()
        newObs = torch.zeros([100, 6])

        newObs[:obs[0], :6] = obs.xyxy[0]
        return newObs


    def getObservation(self):
        frame = ImageGrab.grab()
        outputAttributes = self.attributes
        Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
        crop_minions = frame.crop((Mx1, My1, Mx2, My2))
        Kx1, Ky1, Kx2, Ky2 = 1663, 0, 1718, 25
        crop_kda = frame.crop((Kx1, Ky1, Kx2, Ky2))

        info = [crop_minions, crop_kda]
        for count, i in enumerate(info):
            image = i
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
                    outputAttributes[0] = int(output.strip())
                except:
                    outputAttributes[0] = self.attributes[0]
            elif count == 1:
                try:
                    kda = output.split('/')
                    outputAttributes[1] = int(kda[0])
                    outputAttributes[2] = int(kda[1])
                    outputAttributes[3] = int(kda[2])
                except:
                    outputAttributes[1] = self.attributes[1]
                    outputAttributes[2] = self.attributes[2]
                    outputAttributes[3] = self.attributes[3]
        # Object detection
        results = self.model(frame)
        return results, outputAttributes

    def reward(self, obs, att):
        inM = att[0]
        inK = att[1]
        inD = att[2]
        inA = att[3]
        reward = 0
        done = False
        object_results = obs.pandas().xyxy[0]
        count = 0

        try:
            count += object_results['name'].value_counts()['red_melee']
        except:
            count += 0

        try:
            count += object_results['name'].value_counts()['red_ranged']
        except:
            count += 0

        try:
            count += object_results['name'].value_counts()['red_siege']
        except:
            count += 0

        try:
            count += object_results['name'].value_counts()['red_super']
        except:
            count += 0

        reward += count * 0.5

        if inM > self.attributes[0]:
            reward += inM * 20
            self.attributes[0] = inM
        if inK > self.attributes[1]:
            reward += inK * 300
            self.attributes[1] = inK
        if inD > self.attributes[2]:
            reward += -300
            self.attributes[2] = inD
            done = True
        if inA > self.attributes[3]:
            reward += 80
            self.attributes[3] = inA

        return reward, done
