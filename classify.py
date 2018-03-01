import numpy as np
import os
import cv2
from osgeo import gdal
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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
    nbr_squares = 64
    square_size = 8
    seed_list = []
    max_value = 0


    for m in range(nbr_squares):
        for k in range(nbr_squares):
            next = False
            for i in range(square_size):
                for j in range(square_size):
                    if not next:
                        counter = 0
                        if probs[m*square_size + i][k*square_size + j] > max_value and probs[m*square_size + i][k*square_size + j] > 0.99:
                            next = False
                            max_value = probs[m * square_size + i][k * square_size + j]
                            max_pos_x = m * square_size + i
                            max_pos_y = k * square_size + j
                            pixel = [max_pos_x, max_pos_y]
                            for s in range(-1, 2):
                                for t in range(-1, 2):
                                    x = max_pos_x + s
                                    y = max_pos_y + t
                                    if (x == -1 or y == -1) or (y == 0 and x == 0):
                                        print("do")
                                    else:
                                        if probs[x][y] > 0.99:
                                            counter += 1
                        if counter >= 3:
                            seed_list.append(pixel)
                            max_value = 0
                            next = True
    print(seed_list)
    print(len(seed_list))
    return seed_list


def water_probabilities(prob):
    water_probs = [[0 for x in range(512)] for y in range(512)]
    for i in range(512):
        for j in range(512):
            water_probs[i][j] = prob[i * 512 + j][1]
    return extract_seeds(water_probs)


"""raster_data_path = "fields/image/2298119ene2016recorteTT.tif"
output_fname = "fields/output_image.tiff"
train_data_path = "fields/train"
validation_data_path = "fields/test"

raster_data_path = "island/image/capehorn.tif"
output_fname = "island/output_image.tiff"
train_data_path = "island/train"
validation_data_path = "island/test"""

"""raster_data_path = "landsat_coastline/image/6.tif"
output_fname = "landsat_coastline/output_image_lr.tiff"
train_data_path = "landsat_coastline/train"
#validation_data_path = "landsat_coastline/test"""

"""raster_data_path = "skane_harbour/image/skane_harbour.tif"
output_fname = "skane_harbour/output_classifier.tiff"
train_data_path = "skane_harbour/train"
validation_data_path = "skane_harbour/test"
# blurred_image_path = "capehorn_landpolygons/image/capehorn_blurred.tif"""

raster_data_path = "skane/image/skane.tif"
output_fname = "skane/output_classifier.tiff"
train_data_path = "skane/train"
validation_data_path = "skane/test"

# blurred_image = cv2.GaussianBlur(cv2.imread(raster_data_path),(9,9),0)
# cv2.imshow('Blurred image', blurred_image)
# cv2.imwrite(blurred_image_path, blurred_image)
raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)

# extract_grid()
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()
bands_data = []
for b in range(1, raster_dataset.RasterCount + 1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

bands_data = np.dstack(bands_data)
rows, cols, n_bands = bands_data.shape
files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_data_path, f)
              for f in files if f.endswith('.shp')]

labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
is_train = np.nonzero(labeled_pixels)
training_labels = labeled_pixels[is_train]
training_samples = bands_data[is_train]
"""classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)"""
# classifier = svm.SVC(C=1, kernel='linear', probability=True)
# classifier = GaussianNB()
# classifier = MLPClassifier(alpha=1)
# classifier = LogisticRegression()
classifier = RandomForestClassifier(n_jobs=4, n_estimators=10)
classifier.fit(training_samples, training_labels)

n_samples = rows * cols
flat_pixels = bands_data.reshape((n_samples, n_bands))
result = classifier.predict(flat_pixels)

prob = classifier.predict_proba(flat_pixels)
seed_list = water_probabilities(prob)
classification = result.reshape((rows, cols))
write_geotiff(output_fname, classification, geo_transform, proj)

# ----------Validation----------

"""shapefiles = [os.path.join(validation_data_path, "%s.shp"%c) for c in classes]
print(shapefiles)
verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
for_verification = np.nonzero(verification_pixels)
verification_labels = verification_pixels[for_verification]
predicted_labels = classification[for_verification]

print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(verification_labels, predicted_labels))
target_names = ['Class %s' % s for s in classes]
print("Classification report:\n%s" %
      metrics.classification_report(verification_labels, predicted_labels,
                                    target_names=target_names))
print("Classification accuracy: %f" %
      metrics.accuracy_score(verification_labels, predicted_labels))"""
