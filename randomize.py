import os
import shutil
import random


def randomize_data(input_path, training_path, test_path):
    for image in os.listdir(input_path):
        if image.endswith(".tif"):
            nbr = random.randint(1, 101)
            if nbr <= 20:
                shutil.copyfile(input_path + "/" + image, test_path + "/" + image)
            else:
                shutil.copyfile(input_path + "/" + image, training_path + "/" + image)

input_path = "C:/Users/therese/Google Drive/Kartdata_set/all"
training_path = "C:/Users/therese/Google Drive/Kartdata_set/train"
test_path = "C:/Users/therese/Google Drive/Kartdata_set/test"

randomize_data(input_path, training_path, test_path)
