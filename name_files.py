import os
import shutil


def mix(input_path, output_path):
    print(os.listdir(input_path))
    for folder in os.listdir(input_path):
        print(folder)
        image_path = input_path + "/" + folder
        for image in os.listdir(image_path):
            if image.endswith('.tif'):
                file_path = output_path + "/" + folder + "_" + image
                shutil.copy(image_path + "/" + image, file_path)


input_path = "C:/Users/therese/Google Drive/Kartdata"
output_path = "C:/Users/therese/Google Drive/Kartdata_utan_mappar"

mix(input_path, output_path)
