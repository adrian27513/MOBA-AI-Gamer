import torch
from gym import spaces
import numpy as np
import time
import random
import torch.nn.functional as F
import torch.nn as nn


x = torch.rand(1,4)
# y = torch.rand(2,4)
# print(torch.stack((x,y)))
# print(x)
z = torch.zeros(10,4)
# print(z)
print(x.shape)
z[:list(x.size())[0],:] = x
print(z)
# print(x)
# print(y)
# # print(torch.unsqueeze(x,1))
# # z = torch.cat((x,y))
# # print(z)
# z = torch.concat((x,y))
# print(z)
# # print(torch.stack((x,y)))
# w = torch.stack((z,z))
# print(w)
# print(w.size())
# # print("---")

# print(torch.rand((4,10,5)))



# x = torch.rand((10,6))
# print(x)
# print(x[2:])
# y = torch.rand(1,7)
# y = torch.tensor([-33.50195,  23.38851,   1.07212,  13.41279])
# print(y)
# print(torch.max(y))
# print(y.max(0)[1].item())
# print(torch.tensor([y[0][0].item(),y[0][1].item(),y[0][2].item(),y[0][3].item()]))
# print('---')
# print(torch.mean(x,0,True))
# print(torch.mean(x,0))

# class DQN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(6,12)
#         self.l2 = nn.Linear(12,7)
#
#     def forward(self, X):
#         X = self.l1(X)
#         X = F.relu(X)
#         X = self.l2(X)
#         X = torch.mean(X,0)
#         return X

# net = DQN()
# print(net(x))
# print(net(x)[:2])
# print(net(x)[:2][0].item())

# one = np.ones((5,2))
# obsShape = list(one.shape)
# print(obsShape)
# print((one.shape))
# t1 = time.time()
# time.sleep(5)
# t2 = time.time()
# print(t2-t1)
# print(torch.full((2,3), 5))
#
# print(5 == 4 or 3 or 2)
# print(5 == 5)

# space = spaces.MultiDiscrete([6,1920,1080])
# output = space.sample()
#
# print(output[0])
# print(output[2])
# a = torch.zeros(8, 5)
# print(a)
# #
# b = torch.rand(2,3)
# print(b)
# list1 = list(b.shape)
# print(list1)
#
# a[:list1[0], :3] = b
# print(a)
