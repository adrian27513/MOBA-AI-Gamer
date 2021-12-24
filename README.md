# LeagueHumanPlayerAI
LeagueHumanPlayerAI to train a model to play multiplayer video games (League of Legends) as human players. It is my personal deep dive project into machine learning topics in computer vision, neural networks model building, and reinforcement learning.  

This project was inspired by Oliver Struckmeier’s [LeagueAI project](https://github.com/Oleffa/LeagueAI) and uses his automated training data methodology outlined in his [paper](https://arxiv.org/pdf/1905.13546.pdf) on the subject.

## Method
LeagueHumanPlayerAI simulates how a human would interact and play with League of Legends through analyzing a screen to identify game objects and make basic strategy decisions and perform actions based on the information. To do this LeagueHumanPlayerAI combines:
- **Object Detection** | [YoloV5](https://github.com/ultralytics/yolov5)
- **Optical Character Recognition** | [Tesseract/TesserOCR](https://github.com/sirfz/tesserocr)
- **Reinforcement Learning** | [OpenAI Gym](https://github.com/openai/gym) & [PFRL](https://github.com/pfnet/pfrl)

## Goal
* Automate object detection data collection and generation using OpenCV, PyAutoGUI, and PIL
* Train an object detection model to extract metadata from a screen output using YoloV5 and PyTorch
* Train a Deep Q-Learning algorithm using metadata gained from YoloV5 and Tesseract OCR

### Tasks To Be Accomplished
- [x] Object Detection **(Working)**
- [x] Attribute Observation **(Working)**
- [ ] Gameplay Learning **(Work in Progress)**
- [ ] Team Play **(Future Work)**

## Object Detection
To detect objects on screen LeagueHumanPlayerAI uses YoloV5 to identify and locate 20 objects on screen.
- **mAP@0.5:** 0.83
- **mAP@[0.5:0.95]:** 0.61

### Detectable Objects:
1. Ezreal
2. Ezreal Dead
3. Red Tower
4. Red Melee Minion
5. Red Melee Minion Dead
6. Red Ranged Minion
7. Red Ranged Minion Dead
8. Red Siege Minion
9. Red Siege Minion Dead
10. Red Super Minion
11. Red Super Minion Dead
12. Blue Tower
13. Blue Melee Minion
14. Blue Melee Minion Dead
15. Blue Ranged Minion
16. Blue Ranged Minion Dead
17. Blue Siege Mininon
18. Blue Siege Minion Dead
19. Blue Super Minion
20. Blue Super Minion Dead

### Dataset Generation:
3D models of each of the 11 unique objects and its animations were extracted from the League of Legends client using [LeagueBulkConvert](https://github.com/Jochem-W/LeagueBulkConvert). Due to an animation bug with the red ranged minion model, both blue and red ranged minions are the same model (blue ranged minion) with different skin colors. 

Once the 3D models were obtained, a video was recorded of each model rotating with a viewpoint approximately the same as in game (~55 degrees above front view). Then using modified versions of Oliver Struckmeirer's data generation code, generated dataset images were generated for object detection training.
### TODO:
- Add more objects
  - Champions and related pets
  - Jungle Monsters
  - Inhibitors (Red and Blue)
  - Nexus (Red and Blue)
- Improve Dataset Generation Quality
- Increase Object Detection Accuracy

## Attribute Observation
Cropping the screen and using Tesseract, attribute information like minion count and kills can be tracked.

### Tracked Attributes
1. Minion Count
2. Kills
3. Deaths
4. Assists

### TODO:
- Add more attributes
  - Health
  - Mana
  - Gold
  - Level
  - Selected target's health

## Gameplay Learning
Currently a work in progress creating a reinforcement learning model
- **Plan:** Use Deep Q-Learning to train an intelligent agent to play League of Legends

### Hard-Coded Gameplay Logic
A combination of both object detection and attribute observation to attack and use a random ability on the first red melee minion seen is working.

### TODO:
- Create a Deep Q-Learning algorithm
- Train an agent to play

## Team Play
TODO
