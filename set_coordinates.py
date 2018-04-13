import os
import gdal


def set_coordinates(input_path):
    for folder in os.listdir(input_path):
        image_path = input_path + "/" + folder
        for image in os.listdir(image_path):
            print(image)
            if image.endswith(".tif"):
                factor = 4
                x = image.split('.')[0].split("_")[0]
                y = image.split('.')[0].split("_")[1]

                x_start = 2 ** factor * int(x)
                y_stop = 2 ** factor * int(y)
                y_start = y_stop + 2 ** factor - 1
                x_stop = x_start + 2 ** factor - 1

                src1 = gdal.Open("//10.10.0.94/maps/s2maps/13/" + str(4095 - y_start) + "/" + str(x_start) + ".tif")

                minx1, xres1, xskew1, maxy1, yskew1, yres1 = src1.GetGeoTransform()

                minx = minx1
                maxy = maxy1
                print(image_path + "/" + image + ".tif")
                image_ds = gdal.Open(image_path + "/" + image, gdal.GA_Update)
                geo_transform = minx, xres1*16, xskew1, maxy, yskew1, yres1*16

                image_ds.SetGeoTransform(geo_transform)

set_coordinates("C:/Users/therese/Google Drive/Kartdata_set")
