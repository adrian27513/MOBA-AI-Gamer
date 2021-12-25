from PIL import Image, ImageFilter
import os
from os import listdir
import time


mypath = "E:\\Adrian\\LeagueAI\\RecordingFrames"
champDirectory = sorted(listdir(mypath))
print(champDirectory)

for i in champDirectory:
    ################# Parameters ########################
    number = 0
    print(mypath +"\\"+ i)
    # Input directory of images
    input_dir = mypath +"\\"+ i
    # Output directory of masked and cropped images
    #output_dir = "C:/Users/Adrian/PycharmProjects/MOBAAIGamer/generate_dataset/masked_champ//" + champ + "//" + i
    output_dir = "E:\\Adrian\\LeagueAI\\MaskedChamps\\" + i
    print("Creating new directory for output: {}".format(output_dir))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Area to pre-crop the images to (min_x, min_y, max_x, max_y), can save runtime for large screenshots with small objects
    # Teemo model viewer:
    #area = (700, 300, 1240, 780)
    area = (450, 300, 1450, 950)
    # Greenscreen:
    # x_off = 0
    # y_off = 140
    # area = (0+x_off,0+y_off,1920-x_off,1080-y_off)
    # Background color in RGB
    # Teemo model viewer pink background
    #background = (30, 255,30)
    #background = (95, 80, 170)
    #background = (255,10,10)
    #Windows 3D model viewer
    background = (215, 215, 215)
    background2 = (200, 200, 200)
    # The threshold for removing the background color
    # Greenscreen: 70
    #Purple: 17
    #tolerance = 65
    tolerance = 1
    #tolerance = 100
    # This is needed because there is another shade of pink in the background
    tolerance_offset_1 = 25.0
    tolerance_offset_2 = 25.0  # Greenscreen: 0.74
    tolerance_offset_3 = 25.0  # Teemo viewer: 2.5
    # Remove the outline of the images in case there are any masking artifacts, set the number of layers to be removed
    remove_outline = 0
    #####################################################
    """
    This function applies a filter to a masked image and can either remove 
    or add differntly colored borders around objects
    """
    def modify_outline(f, thickness):
        image = Image.open(output_dir + "/" + f + ".png")
        image = image.convert("RGBA")
        for t in range(thickness):
            mask = image.filter(ImageFilter.FIND_EDGES)
            mask_data = mask.getdata()
            image_data = image.getdata()
            w, h = mask_data.size

            out_data = []
            for y in range(0, h):
                for x in range(0, w):
                    index = x + w * y
                    pixel = (0, 0, 0, 0)
                    if mask_data[index][3] > 0:
                        pixel = (255, 255, 255, 0)
                    else:
                        pixel = (image_data[index][0], image_data[index][1], image_data[index][2], image_data[index][3])
                    out_data.append(pixel)
            image.putdata(out_data)
        image.save(output_dir + "/" + f + ".png", "PNG")


    def get_min_max_x(newData, w, h):
        min_value = 0
        max_value = 0
        for x in range(w - 1, 0, -1):
            for y in range(0, h - 1):
                data_index = x + w * y
                if newData[data_index][3] != 0:
                    max_value = x
                    break
            else:
                continue
            break
        for x in range(0, w - 1):
            for y in range(0, h - 1):
                data_index = x + w * y
                if newData[data_index][3] != 0:
                    min_value = x
                    break
            else:
                continue
            break
        return min_value, max_value


    def get_min_max(newData, w, h):
        min_value = 0
        max_value = 0
        for y in range(h - 1, 0, -1):
            for x in range(0, w - 1):
                data_index = x + w * y
                if newData[data_index][3] != 0:
                    max_value = y
                    break
            else:
                continue
            break
        for y in range(0, h - 1):
            for x in range(0, w - 1):
                data_index = x + w * y
                if newData[data_index][3] != 0:
                    min_value = y
                    break
            else:
                continue
            break
        return min_value, max_value


    # Get list of files in the input directory
    files = sorted(listdir(input_dir))

    for idx, f in enumerate(files):

        fname = f.split(".")[0]
        # Remove the jpg ending
        print(str(int(fname) + number))
        img = Image.open(input_dir + "\\" + fname + ".png")
        # Add alpha channel
        img = img.convert("RGBA")

        # Pre crop to save runtime
        cropped = img.crop(area)
        datas = cropped.getdata()
        newData = []

        for item in datas:
            if background[0] - tolerance_offset_1 < item[0] < background[0] + tolerance_offset_1 \
                    and background[1] - tolerance_offset_2 < item[1] < background[1] + tolerance_offset_2 \
                    and background[2] - tolerance_offset_3 < item[2] < background[2] + tolerance_offset_3:
                newData.append((255, 255, 255, 0))
            # elif background2[0] - tolerance_offset_1 < item[0] < background2[0] + tolerance_offset_1 \
            #         and background2[1] - tolerance_offset_2 < item[1] < background2[1] + tolerance_offset_2 \
            #         and background2[2] - tolerance_offset_3 < item[2] < background2[2] + tolerance_offset_3:
            #     newData.append((255, 255, 255, 0))
            else:
                newData.append((item[0], item[1], item[2], 255))

        cropped.putdata(newData)
        w, h = cropped.size
        # Crop image to pixel content
        min_y, max_y = get_min_max(newData, w, h)
        # Trick: rotate the image by 9 degrees and apply the same functions
        temp = cropped.rotate(90)
        temp_w, temp_h = temp.size
        tempData = temp.getdata()
        min_x, max_x = get_min_max_x(newData, w, h)
        # Save output image as png
        cropped = cropped.crop((min_x, min_y, max_x, max_y))
        cropped.save(output_dir + "/" + str(int(fname) + number) + ".png", "PNG")
        # Remove the outer layer of pixels to remove artifacts from cropping
        modify_outline(str(int(fname) + number), remove_outline)

