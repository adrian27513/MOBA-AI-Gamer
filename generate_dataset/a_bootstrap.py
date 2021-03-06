#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

import numpy as np
from PIL import Image
from PIL import ImageFilter
from os import listdir
import random

####### Object Classes ####
#See LeagueAi.labels

####### Params ############
# Print out the status messages and where stuff is placed
verbose = False
# Important the leaf directories have to be called masked_champions, redMelee, masked_towers or you have to change add_object to write the object classes properly
# Directory in which the masked object images are located
masked_images_dir = "MaskedChamps"
# Directory in which the masked minion images are located
masked_minions = "MaskedMinions"
# Directory in which the map backgrounds are located
map_imags_dir = "map"
# Directory in which the map backgrounds with fog of war are located
map_fog_dir = "map"
# Directory in which the tower images are located
tower_dir = "MaskedTowers"
# Directory in which the inhibs are located
# inhib_dir = "masked_inhib"
# Directory in which the nexuses are located
# nexus_dir = "masked_nexus"
#Directory in which the health_bars are located
# health_bar_path = "health_bars"
#Directory in which abilities are located
# ability_path = "abilities"
# Directory in which the dataset will be stored (creates jpegs and labels subdirectory there)
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
# output_dir = "E:\\Adrian\\LeagueAI\\output"
output_dir = "output"
# Prints a box around the placed object in red (for debug purposes)
print_box = False
# Size of the datasets the program should generate
dataset_size = 4000
# Beginning index for naming output files
start_index = 0
# How many characters should be added minimum/maximum to each sample
characters_min = 1
characters_max = 2
assert (characters_min <= characters_max), "Error, characters_max needs to be larger than minions_min!"
# How many minions should be added minimum/maximum to each sample
minions_min = 3
minions_max = 7
assert (minions_min <= minions_max), "Error, minions_max needs to be larger than minions_min!"
# How many towers should be added to each example
towers_min = 0
towers_max = 2
assert (towers_min <= towers_max), "Error, towers_max needs to be larger than towers_min!"
# The scale factor of how much a champion image needs to be scaled to have a realistic size
# Also you can set a random factor to create more diverse images
scale_champions = 1.1 # 1.0 good
random_scale_champions = 0.2 # 0.12 is good
scale_minions = 0.6 #0.7 good
random_scale_minions = 0.2 # 0.25 good
scale_towers = 2 # 1.6 good
random_scale_towers = 0.2 # 0.2 good
# Random rotation maximum offset in counter-/clockwise direction
rotate = 10
# Make champions seethrough sometimes to simulate them being in a brush, value in percent chance a champion will be seethrough
seethrough_prob = 10
# Output image size
output_size = (1920,1080)
# Factor how close the objects should be clustered around the bias point larger->less clustered but also more out of the image bounds, value around 100 -> very clustered
bias_strength = 210 # 220 is good, dont select too large or the objects will be too often out of bounds
# Resampling method of the object scaling
#sampling_method = Image.BICUBIC
sampling_method = Image.BILINEAR #IMO the best but use both to have more different methods
# Add random noise to pixels
noise = (0,0,0)
# Blur the image
blur = True
blur_strength = 0.6 # 0.6 is a good value
# Sometimes randomly add the overlay
overlay_chance = 5
overlay_path = "ui"
# Add champion icons to overlay
icon_path = "champion_icons"
# Sometimes randomly add a cursor
cursors_min = 0
cursors_max = 3
assert (cursors_min <= cursors_max), "Error, cursors_max needs to be larger than cursors_min!"
cursor_scale = 0.4 #0.45 seems good
cursor_random = 0.2
cursor_path = "cursor"
# Probability of adding a fog of war screenshot with no objects in it
fog_of_war_prob = 0
# Padding for the bias point (to keep the clustering of the minions from spawning minions outside of the image
padding = 400 # 400 is good
########### Helper functions ###################
"""
This funciton applies random noise to the rgb values of a pixel (R,G,B)
"""
def apply_noise(pixel):
    R = max(0, min(255, pixel[0] + random.randint(-noise[0], noise[0])))
    G = max(0, min(255, pixel[1] + random.randint(-noise[1], noise[1])))
    B = max(0, min(255, pixel[2] + random.randint(-noise[2], noise[2])))
    A = pixel[3]
    return (R, G, B, A)
"""
This function places a masked image with a given path onto a map fragment
Passing -1 to the object class allows you to set objects like the UI that are not affected by rotations bias etc.
"""
def add_object(path, cur_image_path, object_class, bias_point, last):
    # Set up the map data
    map_image = Image.open(cur_image_path)
    map_image = map_image.convert("RGBA")
    # Cut the image to the desired output image size
    map_data = map_image.getdata()
    w, h = map_image.size
    if verbose: 
        print("Adding object: ", path)
    # Read the image file of the current object to add
    obj = Image.open(path)
    if object_class >= 0:
        # Randomly rotate the image, but make the normal orientation most likely using a normal distribution
        obj = obj.rotate(np.random.normal(loc=0.0, scale=rotate), expand=True)
    obj = obj.convert("RGBA")
    obj_w, obj_h = obj.size
    # Rescale the image based on the scale factor
    tower_label = (2,11)
    minion_label = (3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19)
    champion_label = (0,1)
    if object_class in tower_label: #towers
        scale_factor = random.uniform(scale_towers-random_scale_towers, scale_towers+random_scale_towers)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    elif object_class in minion_label: #minions
        scale_factor = random.uniform(scale_minions-random_scale_minions, scale_minions+random_scale_minions)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    elif object_class in champion_label: #champions
        scale_factor = random.uniform(scale_champions-random_scale_champions, scale_champions+random_scale_champions)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    elif object_class == -2: # Cursor
        scale_factor = random.uniform(cursor_scale-cursor_random, cursor_scale+cursor_random)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    else:
        size = int(obj_w), int(obj_h)


    # If the object is a champion make it seethrough sometimes to simulate it being in a brush
    in_brush = False
    if object_class >= 0 and np.random.randint(0,100) > 100-seethrough_prob:
        in_brush = True

    # Compute the position of minions based on the bias point. Normally distribute the mininons around 
    # a central point to create clusters of objects for more realistic screenshot fakes
    # Champions and structures are uniformly distributed
    if (object_class in champion_label) or (object_class in tower_label) or (object_class == -2): #Champion or structure or cursor
        obj_pos_center = (random.randint(0, w-1), random.randint(0, h-1))
    else:
        x_coord = np.random.normal(loc=bias_point[0], scale=bias_strength)
        y_coord = np.random.normal(loc=bias_point[1], scale=bias_strength)
        if x_coord < 0:
            x_coord = 0
        if y_coord < 0:
            y_coord = 0
        if x_coord > w:
            x_coord = w
        if y_coord > h:
            y_coord = h
        obj_pos_center = (int(x_coord), int(y_coord))

    # Catch the -1 object class exception to add for example the overlay that has to be centered
    if object_class == -1: # overlay
        obj_pos_center = (int(w/2), int(h/2))
    if object_class == -3: # champion icon in overlay
        # This looks hardcoded but actually it should stay the same ratio
        # for different resolutions as well 
        # Place the champion icon
        obj_pos_center = int(w*(657/1920.0)), int(h*(1015/1080.0))
        size = int(obj_w*0.1), int(obj_h*0.1)
    # Resize the image based on the scaling above
    obj = obj.resize(size, resample=sampling_method)
    obj_w, obj_h = obj.size


    if verbose:
        print("Placing at : {}|{}".format(obj_pos_center[0], obj_pos_center[1]))
    # Extract the image data
    obj_data = obj.getdata()
    out_data = np.array(map_image)
    last_pixel = 0
    # Compute the object corners
    min_x = int(min(w, max(0, obj_pos_center[0] - obj_w/2 - 2)))
    max_x = int(min(w, max(0, obj_pos_center[0] + obj_w/2 + 2)))
    min_y = int(min(h, max(0, obj_pos_center[1] - obj_h/2 - 2)))
    max_y = int(min(h, max(0, obj_pos_center[1] + obj_h/2 + 2)))
    # Place the images
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            pixel = (0, 0, 0, 0)
            # Compute the pixel index in the map fragment
            map_index = x + w * y
            #print("x: ", x, " y: ", y)
            # If we want to print the box around the object, set the pixel to red
            if print_box is True and y == obj_pos_center[1] - int(obj_h / 2) and \
                    obj_pos_center[0] - int(obj_w / 2) < x < obj_pos_center[0] + int(obj_w / 2):
                pixel = (255, 0 ,0, 255)
            elif print_box is True and y == obj_pos_center[1]+int(obj_h / 2) and \
                    obj_pos_center[0] - int(obj_w / 2) < x < obj_pos_center[0] + int(obj_w / 2):
                pixel = (255,0,0,255)
            elif print_box is True and x == obj_pos_center[0]-int(obj_w / 2) and \
                    obj_pos_center[1] - int(obj_h / 2) < y < obj_pos_center[1] + int(obj_h / 2):
                pixel = (255,0,0,255)
            elif print_box is True and x == obj_pos_center[0]+int(obj_w / 2) and \
                    obj_pos_center[1] - int(obj_h / 2) < y < obj_pos_center[1] + int(obj_h / 2):
                pixel = (255,0,0,255)
            else:
                # Replace the old input image pixels with the object to add pixels
                if obj_pos_center[0] - int(obj_w / 2) <= x <= obj_pos_center[0] + int(obj_w / 2) \
                        and obj_pos_center[1] - int(obj_h / 2) <= y <= obj_pos_center[1] + int(obj_h / 2):
                    obj_x = x - obj_pos_center[0] - int(obj_w / 2) -1
                    obj_y = y - obj_pos_center[1] - int(obj_h / 2)-1
                    object_index = (obj_x + obj_w * obj_y)
                    # Check the alpha channel of the object to add
                    # If it is smaller 150, the pixel is invisible, 255: fully visible, 150: seethrough (brush simulation)
                    # Then use the original images pixel value
                    # Else use the object to adds pixel value
                    if obj_data[object_index][3] == 255:
                        if in_brush and last_pixel % 3==0:
                            # take the map pixel every second time to make the champion seethrough
                            pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
                            last_pixel += 1
                        else:
                            pixel = (obj_data[object_index][0],  obj_data[object_index][1], obj_data[object_index][2], 255)
                            last_pixel += 1
                    elif obj_data[object_index][3] == 0:
                        pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], 255)
                else:
                    pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
            out_data[y,x] = pixel
    if last and (noise[0] > 0 or noise[1] > 0 or noise[2] > 0):
        for y in range(0, h):
            for x in range(0, w):
                # Compute the pixel index in the map fragment
                map_index = x + w * y
                out_data[y,x] = apply_noise(out_data[y,x])
    # Save the image
    map_image = Image.fromarray(np.array(out_data))
    if blur and last:
        map_image = map_image.filter(ImageFilter.GaussianBlur(radius=blur_strength))
    map_image = map_image.convert("RGB")
    map_image.save(output_dir+"/images/"+filename+".jpg", "JPEG")
    # Append the bounding box data to the labels file if the object class is not -1
    if object_class >= 0:
        with open(output_dir+"/labels/"+filename+".txt", "a") as f:
            # Write the position of the object and its bounding box data to the labels file
            # All values are relative to the whole image size
            # Format: class, x_pos, y_pos, width, height
            object_x_pos = float(obj_pos_center[0]/w)
            object_y_pos = float(obj_pos_center[1]/h)
            object_width = float(obj_w/w)
            object_height = float(obj_h/h)

            if object_x_pos - (object_width/2.0) < 0:
                object_width = object_x_pos + (object_width / 2.0)
                object_x_pos = (object_width / 2.0)
            if object_x_pos + (object_width / 2.0) > 1:
                object_width = 1 - (object_x_pos - (object_width / 2.0))
                object_x_pos = 1 - (object_width / 2.0)

            if object_y_pos - (object_height/2.0) < 0:
                object_height = object_y_pos + (object_height / 2.0)
                object_y_pos = (object_height / 2.0)
            if object_y_pos + (object_height / 2.0) > 1:
                object_height = 1 - (object_y_pos - (object_height / 2.0))
                object_y_pos = 1 - (object_height / 2.0)

            f.write("" + str(object_class) + " " + str(object_x_pos) + " " + str(object_y_pos) + " " + str(object_width) + " " + str(object_height) + "\n")


########### Main function ######################
obj_dirs = sorted(listdir(masked_images_dir))
maps = sorted(listdir(map_imags_dir))

for dataset in range(0, dataset_size):
    filename = str(dataset+start_index)
    print("Dataset: ", dataset, " // ", dataset_size, " : ", filename)
    # Randomly select a map background
    mp_fnam = map_imags_dir+"//"+random.choice(maps)
    if verbose:
        print("Using map fragment: ", mp_fnam)

    # Randomly select a set of characters to add to the image
    characters = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        randomNumber = random.random()
        if randomNumber < 0.2:
            temp_obj_folder = "ezreal_dead"
        else:
            temp_obj_folder = "ezreal"
        temp_obj_path = masked_images_dir+"\\"+temp_obj_folder+"\\"
        # Select a random masked image of that object
        if temp_obj_folder == "ezreal":
            characters.append([temp_obj_path+"/"+random.choice(sorted(listdir(temp_obj_path))),0])
        elif temp_obj_folder == "ezreal_dead":
            characters.append([temp_obj_path+random.choice(sorted(listdir(temp_obj_path))),1])
    if verbose: 
        print("Adding {} champions!".format(len(characters)))

    # Randomly add 0-12 minions to the image
    minions = []
    for i in range(0, random.randint(minions_min, minions_max)):
        # Select a random subdirectory because the minions are sorted in subdirectories
        minions_dir = random.choice(sorted(listdir(masked_minions)))
        minion_dir2 = masked_minions+"\\"+minions_dir
        minion_loc = minion_dir2+"\\"+random.choice(sorted(listdir(minion_dir2)))
        if minions_dir == "red_siege":
            minions.append([minion_loc, 3])
        elif minions_dir == "red_siege_dead":
            minions.append([minion_loc, 4])
        elif minions_dir == "red_ranged":
            minions.append([minion_loc, 5])
        elif minions_dir == "red_ranged_dead":
            minions.append([minion_loc, 6])
        elif minions_dir == "red_melee":
            minions.append([minion_loc, 7])
        elif minions_dir == "red_melee_dead":
            minions.append([minion_loc, 8])
        elif minions_dir == "red_super":
            minions.append([minion_loc, 9])
        elif minions_dir == "red_super_dead":
            minions.append([minion_loc, 10])
        elif minions_dir == "blue_siege":
            minions.append([minion_loc, 12])
        elif minions_dir == "blue_siege_dead":
            minions.append([minion_loc, 13])
        elif minions_dir == "blue_ranged":
            minions.append([minion_loc, 14])
        elif minions_dir == "blue_ranged_dead":
            minions.append([minion_loc, 15])
        elif minions_dir == "blue_melee":
            minions.append([minion_loc, 16])
        elif minions_dir == "blue_melee_dead":
            minions.append([minion_loc, 17])
        elif minions_dir == "blue_super":
            minions.append([minion_loc, 18])
        elif minions_dir == "blue_super_dead":
            minions.append([minion_loc, 19])
        else:
            print("Error: This folder: ", minions_dir, " was not specified to contain masked images. Skipping. Atention! Dataset might be broken!")
    if verbose: 
        print("Adding {} minions!".format(len(minions)))

    # Randomly select one tower image
    towers = []
    for i in range(0, random.randint(towers_min, towers_max)):
        # Select a random subdirectory because the towers are sortedy in blue/red team folders
        towers_dir = random.choice(sorted(listdir(tower_dir)))
        if towers_dir == "red":
            towers.append([tower_dir+"/"+towers_dir+"/"+random.choice(sorted(listdir(tower_dir+"/"+towers_dir))), 2])
        if towers_dir == "blue":
            towers.append([tower_dir + "/" + towers_dir + "/" + random.choice(sorted(listdir(tower_dir + "/" + towers_dir))), 11])
    if verbose:
        print("Adding {} towers!".format(len(towers)))

    # Add a fog of war / empty screenshot
    if 100 - fog_of_war_prob < random.randint(0,100):
        fog_file = random.choice(sorted(listdir(map_fog_dir)))
        fog_name = map_fog_dir+"/"+fog_file
        map_image = Image.open(fog_name)
        map_image = map_image.convert("RGB")
        map_image.save(output_dir+"/images/"+filename+".jpg", "JPEG")
        # Save empty label  because we did not place any objects
        with open(output_dir+"/labels/"+filename+".txt", "a") as f:
            f.write("")
    else:
        # Now figure out the order in which we want to add the objects (So that sometimes objects will overlap)
        objects_to_add = characters+minions+towers
        random.shuffle(objects_to_add)
        # Read in the current map background as image
        map_image = Image.open(mp_fnam)
        w, h = map_image.size
        # Make sure the image is 1920x1080 (otherwise the overlay might not fit properly)
        assert (w == 1920 and h == 1080), "Error image has to be 1920x1080"
        map_image = map_image.convert("RGB")
        map_image.save(output_dir+"/images/"+filename+".jpg", "JPEG")
        cur_image_path = output_dir+"/images/"+filename+".jpg"
        # Iterate through all objects in the order we want them to be added and add them to the backgroundl
        # Note this function also saves the image already
        # Point around which the objects will be clustered
        bias_point = (random.randint(padding, w-1-padding), random.randint(padding, h-1-padding))
        # Add the overlay, the bias point plays no role here because of the object class (object class -1 is not added to the labels.txt)
        if random.randint(0,100) > 100 - overlay_chance:
            overlay_name = overlay_path+"/"+random.choice(sorted(listdir(overlay_path)))
            # Add a champion icon if it is the normal ui overlay
            if overlay_name == overlay_path+"/overlay.png":
                add_object(icon_path+"/"+random.choice(sorted(listdir(icon_path))), cur_image_path, -3,bias_point, False)
            add_object(overlay_name, cur_image_path, -1, bias_point, False)
        for i in range(0, len(objects_to_add)):
            o = objects_to_add.pop()
            if len(objects_to_add) == 0:
                add_object(o[0], cur_image_path, o[1], bias_point, True)# Set last to true to apply the possible noise
            else:
                add_object(o[0], cur_image_path, o[1], bias_point, True)
        #Add a cursor
        for i in range(cursors_min,random.randint(cursors_min, cursors_max)):
            cursor = cursor_path+"/"+random.choice(sorted(listdir(cursor_path)))
            add_object(cursor, cur_image_path, -2, bias_point, False)
        # If no objects were added we still have to create an empty txt file
        with open(output_dir+"/labels/"+filename+".txt", "a") as f:
            f.write("")
    if verbose:
        print("=======================================")

