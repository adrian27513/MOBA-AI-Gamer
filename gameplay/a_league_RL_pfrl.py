import gym
from a_league_env import LeagueEnv
import torch.nn
import math
import random
import numpy
import pfrl
import time
import pydirectinput
import pyautogui

def winGame():
    pydirectinput.keyDown('ctrl')
    pydirectinput.keyDown('shift')
    pydirectinput.press('.')
    pydirectinput.keyUp('shift')
    pydirectinput.keyUp('ctrl')

def resetGame():
    pyautogui.leftClick(720, 920)
    time.sleep(1)
    pyautogui.leftClick(315, 115)
    time.sleep(1)
    pyautogui.leftClick(475, 190)
    time.sleep(1)
    pyautogui.leftClick(940, 335)
    time.sleep(1)
    pyautogui.leftClick(835, 930)
    time.sleep(1)
    pyautogui.leftClick(835, 930)
    time.sleep(7)
    pyautogui.leftClick(1275, 750)
    time.sleep(1)
    pyautogui.leftClick(960, 820)
    time.sleep(50)
    pyautogui.mouseDown(button='left', x=110, y=495)
    time.sleep(1)
    pyautogui.mouseUp(button='left')
    time.sleep(0.2)
    pyautogui.mouseDown(button='left', x=110, y=495)
    time.sleep(1)
    pyautogui.mouseUp(button='left')
    pydirectinput.press('a')

env = LeagueEnv()
class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

obs_size = 6
n_actions = 3
q_func = QFunction(obs_size, n_actions)

q_func2 = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

# Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    # replay_start_size=100,
    update_interval=1,
    target_update_interval=100,
    # phi = phi,
    gpu=0
)

n_episodes = 300
max_episode_len = 1000
t_start = time.time()
t1 = time.time()
reset = False
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        t2 = time.time()
        if t2 - t1 >= 3480:
            done = True
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    print('episode:', i, 'R:', R)

    if i % 5 == 0:
        print('statistics:', agent.get_statistics())
        agent.save('agent')
    t3 = time.time()
    if t3 - t1 >= 3480:
        winGame()
        time.sleep(30)
        resetGame()
        time.sleep(1)
        t1 = time.time()
    if t3 - t_start > 43200:
        break
print('Finished.')