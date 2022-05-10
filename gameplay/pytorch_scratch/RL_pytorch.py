import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM
import pyautogui
import pydirectinput
import tesserocr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image, ImageGrab
import time
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# def sigmoid(x):
#
#     if x >= 0:
#         z = math.exp(-x)
#         sig = 1 / (1 + z)
#         return sig
#     else:
#         z = math.exp(x)
#         sig = z / (1 + z)
#         return sig

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda")
# device = 'cuda'

# Minions  Kills  Deaths  Assists
attributes = [0, 0, 0, 0]

api = tesserocr.PyTessBaseAPI()

model = torch.hub.load('C:\\Users\\Adrian\\PycharmProjects\\MOBAAIGamer\\yolov5', 'custom',
                                    path='C:\\Users\\Adrian\\PycharmProjects\\MOBAAIGamer\\LeagueAIWeights.pt',
                                    source='local')
OBS_SIZE = 10
model.conf = 0.6  # NMS confidence threshold
model.iou = 0.5  # NMS IoU threshold
model.classes = [0, 2, 3, 5, 7, 9, 11, 12, 14, 16, 18]  # Remove dead classes
model.multi_label = False  # NMS multiple labels per box
model.max_det = OBS_SIZE  # maximum number of detections per image

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,16), nn.ReLU(), nn.Linear(16,7))

    def forward(self, x):
        x = self.net(x)
        x = torch.mean(x, 0)
        return x


def getObservation():
    frame = ImageGrab.grab()
    outputAttributes = attributes.copy()
    Mx1, My1, Mx2, My2 = 1775, 0, 1820, 30
    crop_minions = frame.crop((Mx1, My1, Mx2, My2))
    Kx1, Ky1, Kx2, Ky2 = 1663, 0, 1718, 25
    crop_kda = frame.crop((Kx1, Ky1, Kx2, Ky2))

    info = [crop_minions, crop_kda]
    for count, i in enumerate(info):
        image = np.array(i)
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
                outputAttributes[0] = attributes[0]
        elif count == 1:
            try:
                kda = output.split('/')
                outputAttributes[1] = int(kda[0])
                outputAttributes[2] = int(kda[1])
                outputAttributes[3] = int(kda[2])
            except:
                outputAttributes[1] = attributes[1]
                outputAttributes[2] = attributes[2]
                outputAttributes[3] = attributes[3]
    # Object detection
    results = model(frame)

    object_results = results.pandas().xyxy[0]
    object_results = object_results.assign(xmid=(object_results['xmin'] + object_results['xmax']) / 2)
    object_results = object_results.assign(ymid=(object_results['ymin'] + object_results['ymax']) / 2)
    object_results = object_results.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'name'])
    results_updated = torch.zeros((10, 4), device=device).float()
    if not results.xyxy[0].shape[0] == 0:
        results_tensor = torch.tensor(object_results.values, device=device).float()
        results_updated[:list(results_tensor.size())[0], :] = results_tensor

    # print(object_results)
    # print(results_tensor)

    return results_updated, outputAttributes


def reset():
    # Reset
    pyautogui.mouseUp(button='right')
    pydirectinput.keyDown('ctrl')
    pydirectinput.keyDown('shift')
    pydirectinput.keyDown('p')
    pydirectinput.keyUp('ctrl')
    pydirectinput.keyUp('shift')
    pydirectinput.keyUp('p')
    pydirectinput.keyDown('shift')
    pydirectinput.press('y', presses=5)
    pydirectinput.press('c')
    pydirectinput.press('u')
    pydirectinput.keyUp('shift')
    pydirectinput.keyDown('ctrl')
    pydirectinput.press('q', presses=3)
    pydirectinput.press('w')
    pydirectinput.press('e')
    pydirectinput.press('r')
    pydirectinput.keyUp('ctrl')

    pyautogui.rightClick(1710, 900)
    pyautogui.mouseUp(button='right')
    time.sleep(85)

    obs, _ = getObservation()
    results_obs = obs
    # if results_obs.shape[0] == 0:
    #     results_obs = torch.zeros((1,4), dtype=torch.float64, device=device)
    return results_obs


def rewards(obs, att):
    inM = att[0]
    inK = att[1]
    inD = att[2]
    inA = att[3]
    reward = 0
    done = False
    object_results = obs
    count = 0

    if inM > attributes[0]:
        reward += inM * 20
        attributes[0] = inM
    if inK > attributes[1]:
        reward += inK * 300
        attributes[1] = inK
    if inD > attributes[2]:
        done = True
        reward += -300
        attributes[2] = inD
    if inA > attributes[3]:
        reward += 80
        attributes[3] = inA

    try:
        count += object_results['class'].value_counts()[11] #red melee
    except:
        count += 0

    try:
        count += object_results['class'].value_counts()[13] # red ranged
    except:
        count += 0

    try:
        count += object_results['class'].value_counts()[15] # red siege
    except:
        count += 0

    try:
        count += object_results['class'].value_counts()[17] # red super
    except:
        count += 0

    reward += count * 0.5

    return reward, done


def clamp(n, small, large):
    return max(small, min(n, large))
def step(action):
    # Move Mouse
    print("---")
    # print(state)
    print(action)
    # extracted_action = action[0]
    set_action = action[0].item()

    moveLocX = clamp(action[1].item(), 1, 1919)
    moveLocY = clamp(action[2].item(), 1, 1079)
    pyautogui.moveTo(x=moveLocX, y=moveLocY)

    # Press Buttons
    if set_action == 0:
        # Press Q
        pydirectinput.press('q')
        # print('q')
    elif set_action == 1:
        # Press W
        pydirectinput.press('w')
        # print('w')
    elif set_action == 2:
        # Press E
        pydirectinput.press('e')
        # print('e')
    elif set_action == 3:
        # Press R
        pydirectinput.press('r')
        # print('r')
    elif set_action == 4:
        # Right-Click
        pyautogui.click(button='right')
        # print('click')
        pyautogui.mouseUp(button='right')

    obs, att = getObservation()
    reward, done = rewards(obs, att)

    return reward, done


policy_net = DQN().to(device)
target_net = DQN().to(device)

# policy_net = DQN().cuda()
# target_net = DQN().cuda()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_state = policy_net(state).float().to(device)
            return policy_state, False
    else:
        return torch.rand(7,device=device), True


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # print(batch.action)
    state_batch = torch.stack(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).to(torch.int64).to(device)
    reward_batch = torch.cat(batch.reward).float().to(device)

    # print("===")
    # print(state_batch)
    # print(action_batch)
    # print("===")
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(0, action_batch).to(device)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(0)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    reset()
    state, att = getObservation()
    # print("state")
    # print(state)
    # time.sleep(100)
    zero_minion = 0
    minion_count = 0
    cycles = 0
    for t in count():
        # Select and perform an action
        action, policy = select_action(state)
        policy_output = action
        slice_out = torch.tensor([action[2].item(), action[3].item(), action[4].item(), action[5].item(), action[6].item()], device=device)
        if policy:
            action = torch.tensor([random.randrange(5), random.randrange(1,1920),random.randrange(1,1080)], device=device, dtype=torch.int64)
        else:
            # print(action)
            action = torch.tensor([slice_out.max(0)[1].item(), policy_output[0].item(), policy_output[1].item()], device=device)

        print("t")
        print(t)
        # print("===")
        # print(action)
        # print("===")
        reward, done = step(action)

        minion_diff = attributes[0] - minion_count
        if minion_diff == 0:
            zero_minion = zero_minion + 1
        else:
            zero_minion = 0
            minion_count = attributes[0]

        if zero_minion >= 20:
            reward = reward - 20 - (10 * cycles)
            cycles = cycles + 1

        print("reward")
        print(reward)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state, att = getObservation()
            # print("state")
            # print(next_state)
            # next_state = torch.tensor(next_state, device=device)
            # print(next_state)
        else:
            next_state = None

        action = slice_out.max(0)[1].view(1,1)
        # print(action)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')


