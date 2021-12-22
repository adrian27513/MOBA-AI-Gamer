# This script just needs Opencv to run
# Install opencv using pip: python -m pip install opencv-python
# See this link for more information: https://www.scivision.co/install-opencv-python-windows/
import cv2
import math
import os
import datetime
from os import listdir
from os.path import isfile, join
import time

# champ = "blueSuper"
# #mypath = "E:\LeagueAI\champs\\" + champ + "\default\\video recordings"
# mypath = "E:\LeagueAI\minions\\red\\" + champ
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print (onlyfiles)

recordingPath = "E:\\Adrian\\LeagueAI\\Recordings"
recordingDirectory = sorted(listdir(recordingPath))
print(recordingDirectory)

for champ in recordingDirectory:
    champPath = recordingPath + "\\" + champ
    champDirectory = sorted(listdir(champPath))
    print(champDirectory)
    output_directory = "E:\\Adrian\\LeagueAI\\RecordingFrames\\" + champ + "\\"
    print("Creating new directory for output: {}".format(output_directory))
    os.makedirs(output_directory)
    for i in champDirectory:
        # ======================= Get parameters =========================
        SOURCE_FILENAME = champPath + "\\" + i
        SKIP_FRAMES = 1
        OUT_FILE_PREFIX = ""
        source = cv2.VideoCapture(SOURCE_FILENAME)
        print(
            "Starting " + i + " export, cancel the process by pressing ctrl+c. All images that are already exported will be saved!")
        if (source.isOpened() == False):
            print('Error opening video stream')
        else:
            frame_count = 0
            output_counter = 0 + len(listdir(output_directory))
            print(output_counter)
            while (source.isOpened()):
                ret, frame = source.read()
                if ret:
                    if frame_count % SKIP_FRAMES == 0:
                        # Write to file
                        output_filename = output_directory + OUT_FILE_PREFIX + str(output_counter) + ".png"
                        output_counter = output_counter + 1
                        cv2.imwrite(output_filename, frame)
                    frame_count = frame_count + 1

                else:
                    print('File Processed!')
                    break