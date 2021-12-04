import os
import glob
from os import listdir

dataPath = "C:\\Users\\Adrian\\PycharmProjects\\AdrianLeagueAIYoloV5\\data"
dataFolders = listdir(dataPath)

for d in dataFolders:
    currentPath = dataPath + "\\" + d
    testTrainFolders = listdir(currentPath)
    for t in testTrainFolders:
        currentInnerPath = currentPath + "\\" + t
        currentInnerFiles = listdir(currentInnerPath)
        for file in currentInnerFiles:
            currentFile = currentInnerPath + "\\" + file
            os.remove(currentFile)
            print("Removed " + currentFile)