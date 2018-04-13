import numpy as np
import os
from osgeo import gdal
import pickle
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import re
from morphsnakes_large import start_snake

# A list of "random" colors (for a nicer output)
COLORS = ["#FF0000", "#09780D", "#2930C1"]


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(len(classes) + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def extract_seeds(probs):
    nbr_squares = 16
    square_size = 32
    seed_list = []
    max_value = 0
    for m in range(nbr_squares):
        for k in range(nbr_squares):
            next = False
            max_value = 0
            for i in range(square_size):
                for j in range(square_size):
                    if not next:
                        counter = 0
                        if probs[m*square_size + i][k*square_size + j] > max_value and probs[m*square_size + i][k*square_size + j] > 0.99:
                            max_value = probs[m * square_size + i][k * square_size + j]
                            max_pos_x = m * square_size + i
                            max_pos_y = k * square_size + j
                            pixel = [max_pos_x, max_pos_y]
                            for s in range(-1, 2):
                                for t in range(-1, 2):
                                    x = max_pos_x + s
                                    y = max_pos_y + t
                                    if (x == -1 or y == -1) or (y == 0 and x == 0) or (x == 512 or y == 512):
                                        print("Out of range")
                                    else:
                                        if probs[x][y] > 0.99:
                                            counter += 1
                        if counter >= 3:
                            seed_list.append(pixel)
                            next = True
    return seed_list


def extract_seeds_squares(probs):
    print("Extracting seeds...")
    nbr_squares = 64
    square_size = 8
    seed_list = []
    for m in range(nbr_squares):
        for k in range(nbr_squares):
            next = False
            counter = 0
            for i in range(square_size):
                for j in range(square_size):
                    if not next:
                        if probs[m*square_size + i][k*square_size + j] > 0.6:
                            counter += 1
                        if counter == square_size*square_size:
                            max_pos_x = int(m * square_size + i - square_size/2)
                            max_pos_y = int(k * square_size + j - square_size/2)
                            pixel = [max_pos_x, max_pos_y]
                            seed_list.append(pixel)
                            next = True
                            counter = 0
                            break
    return seed_list


def get_mean(list):
    total = 0
    for item in list:
        print(item)
        total = total + item
    mean = total/len(list)
    return mean

def get_seeds_from_osm(verification_pixels):
    print("Extracting seeds...")
    nbr_squares = 32
    square_size = 16
    seed_list = []
    for m in range(nbr_squares):
        for k in range(nbr_squares):
            next = False
            counter = 0
            for i in range(square_size):
                for j in range(square_size):
                    if not next:
                        x = m * square_size + i
                        y = k * square_size + j
                        if verification_pixels[x][y] == 2:
                            counter += 1
                        if counter == square_size * square_size:
                            x_pos = int(x - square_size / 2)
                            y_pos = int(y - square_size / 2)
                            pixel = [x_pos, y_pos]
                            seed_list.append(pixel)
                            next = True
                            counter = 0
                            break
    return seed_list


def get_median(list):
    list.sort()
    median = int(len(list)/2)
    return list[median]


def water_probabilities(prob):
    water_probs = [[0 for x in range(512)] for y in range(512)]
    for i in range(512):
        for j in range(512):
            water_probs[i][j] = prob[i * 512 + j][1]
    return extract_seeds_squares(water_probs)


def load_training_data(directory, shapefile):
    count = 0
    for image in os.listdir(directory):
        if image.endswith('.tif'):
            count += 1
            raster_image = gdal.Open(os.path.join(directory, image), gdal.GA_ReadOnly)
            geo_transform = raster_image.GetGeoTransform()
            projection = raster_image.GetProjectionRef()
            bands_d = []
            for b in range(1, raster_image.RasterCount):
                band = raster_image.GetRasterBand(b)
                bands_d.append(band.ReadAsArray())

            bands_d = np.dstack(bands_d)
            rows, cols, bands = bands_d.shape
            labeled_pixels = vectors_to_raster(shapefile, rows, cols, geo_transform, projection)
            is_train = np.nonzero(labeled_pixels)
            training_l = labeled_pixels[is_train]
            training_s = bands_d[is_train]

            if count == 2:
                training_l_result = np.concatenate((training_l, prev_l))
                training_s_result = np.concatenate((training_s, prev_s))

            if count >= 3:
                training_l_result = np.concatenate((training_l, training_l_result))
                training_s_result = np.concatenate((training_s, training_s_result))

            prev_l = training_l
            prev_s = training_s

    return training_l_result, training_s_result


def predict_test_data(directory, shapefiles):
    all_num_iters = []
    all_execution_time = []
    all_similarity = []
    all_verification_labels = []
    all_predicted_labels = []
    for image in os.listdir(directory):
        if image.endswith('.tif'):
            print("Image: ", image)
            image_nbr = image.split(".")
            test_image = gdal.Open(os.path.join(directory, image), gdal.GA_ReadOnly)
            geo_transform = test_image.GetGeoTransform()
            projection = test_image.GetProjectionRef()
            test_image_data = []

            for b in range(1, test_image.RasterCount):
                band = test_image.GetRasterBand(b)
                test_image_data.append(band.ReadAsArray())
            test_image_data = np.dstack(test_image_data)

            row, col, n_band = test_image_data.shape

            n_samples = row * col
            flat_pixels = test_image_data.reshape((n_samples, n_band))

            result = loaded_model.predict(flat_pixels)
            classification = result.reshape((row, col))

            prob = loaded_model.predict_proba(flat_pixels)
            list_of_seeds = water_probabilities(prob)

            """Comment this line out if no output image is needed"""
            #write_geotiff(("output_" + str(image_nbr[0]) + ".tiff"), classification, geo_transform, projection)

            verification_pixels = vectors_to_raster(shapefiles, row, col, geo_transform, projection)
            for_verification = np.nonzero(verification_pixels)

            """Extract seed list from classifier"""
            #prob = loaded_model.predict_proba(flat_pixels)
            #list_of_seeds = water_probabilities(prob)
            """Extract seed list from osm truth"""
            list_of_seeds = get_seeds_from_osm(verification_pixels)

            verification_labels = verification_pixels[for_verification]
            predicted_labels = classification[for_verification]

            if len(list_of_seeds) > 0:
                similarity, execution_time, num_iters = start_snake(test_image, str(image_nbr[0]), verification_pixels, list_of_seeds)
                all_execution_time.append(execution_time)
                all_num_iters.append(num_iters)
                all_similarity.append(similarity)

            all_verification_labels = np.concatenate((all_verification_labels, verification_labels))
            all_predicted_labels = np.concatenate((all_predicted_labels, predicted_labels))

    return all_verification_labels, all_predicted_labels, list_of_seeds, all_similarity, all_num_iters, all_execution_time


files = [f for f in os.listdir("Dataset/train") if f.endswith('.shp')]

classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join("Dataset/train", f)
              for f in files if f.endswith('.shp')]


# ----- Train the model -------

"""training_labels, training_samples = load_training_data("set", shapefiles)

classifier = RandomForestClassifier(n_jobs=4, n_estimators=10)
model = classifier.fit(training_samples, training_labels)

filename = 'small_model_3bands.sav'

pickle.dump(model, open(filename, 'wb'))"""


# ------- Predict -----------

loaded_model = pickle.load(open('small_model_3bands.sav', 'rb'))

shapefiles_test = [os.path.join("Dataset/test", "%s.shp"%c) for c in classes]

verification_labels, predicted_labels, seed_list, all_similarity, all_num_iters, all_execution_time = predict_test_data("Dataset/test_images", shapefiles_test)



# -------- Validation --------

print("Mean similarity: ", get_mean(all_similarity))

print("Median: ", get_median(all_similarity))

print("Confusion matrix:\n%s" %
      metrics.confusion_matrix(verification_labels, predicted_labels))
target_names = ['Class %s' % s for s in classes]
print("Classification report:\n%s" %
      metrics.classification_report(verification_labels, predicted_labels,
                                    target_names=target_names))
print("Classification accuracy: %f" %
      metrics.accuracy_score(verification_labels, predicted_labels))
