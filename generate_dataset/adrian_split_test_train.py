#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from os import listdir
import numpy as np
import random
import os
import datetime
from os import listdir
from os.path import isfile, join

# This script is used to split the datasets generated with bootstrap.py into test and training dataset

# Attention if you use darknet to train: the structure has to be exactly as follows:
# - Dataset
# -- images
# --- XYZ0.jpg
# --- XYZ1.jpg
# -- labels
# --- XYZ0.txt
# --- XYZ1.txt
# -- train.txt
# -- test.txt



# Set number of datasets that will be randomly selected for test dataset
split = 1000
# Overall size of the dataset
dataset_size = 2000

# Directory of the dataset (parent directory of jpegs and labels folder)
dataset_path = "C:/Users/Adrian/PycharmProjects/AdrianLeagueAIYoloV5/generate_dataset/output"

# Randomly shuffle the list of samples in the dataset and select random test and train samples
datasets = listdir(dataset_path+"/images/")
random.shuffle(datasets)
datasets_valid = datasets[:split]
datasets_train = datasets[split:]
print(datasets_train)
print(datasets_valid)

for i in range(0,len(datasets_train)):
    os.rename(dataset_path+"/images/"+datasets_train[i],
              "C:/Users/Adrian/PycharmProjects/AdrianLeagueAIYoloV5/data/images/train/"+datasets_train[i])
    os.rename(dataset_path+"/labels/"+datasets_train[i].replace(".jpg",".txt"),
              "C:/Users/Adrian/PycharmProjects/AdrianLeagueAIYoloV5/data/labels/train/"+datasets_train[i].replace(".jpg",".txt"))
    print("Adding: " + datasets_train[i].replace(".jpg", ""))

for i in range(0,len(datasets_valid)):
    os.rename(dataset_path+"/images/"+datasets_valid[i],
              "C:/Users/Adrian/PycharmProjects/AdrianLeagueAIYoloV5/data/images/valid/"+datasets_valid[i])
    os.rename(dataset_path+"/labels/"+datasets_valid[i].replace(".jpg",".txt"),
              "C:/Users/Adrian/PycharmProjects/AdrianLeagueAIYoloV5/data/labels/valid/"+datasets_valid[i].replace(".jpg",".txt"))
    print("Adding: " + datasets_valid[i].replace(".jpg", ""))


