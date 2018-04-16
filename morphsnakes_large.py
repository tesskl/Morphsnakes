# -*- coding: utf-8 -*-

from itertools import cycle
import numpy as np
import time

from scipy.ndimage import binary_dilation, binary_erosion
from osgeo import gdal
import os
from scipy.ndimage import gaussian_filter


# COLORS = ["#008000", "#003cb3"]
COLORS = ["#ffffff", "#000000"]
RASTERIZE_COLOR_FIELD = "__color__"


class fcycle(object):
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0, 1, 0]] * 3), np.flipud(np.eye(3)), np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1

_aux = np.zeros(0)


def SI(u):
    """SI operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)

    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_erosion(u, P_i)

    return _aux.max(0)


def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)

    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_dilation(u, P_i)

    return _aux.min(0)


# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])


class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""

    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        """Create a Morphological ACWE solver.

        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.data = data

    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u > 0] = 1
        self._u[u <= 0] = 0

    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")

    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u

        if u is None:
            raise ValueError("the levelset function is not set (use set_levelset)")

        data = self.data

        # Determine c0 and c1.
        inside = u > 0
        outside = u <= 0

        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())

        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)
        # aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        aux = abs_dres * (self.lambda1 * (data - c1) ** 2 - self.lambda2 * (data - c0) ** 2)

        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0

        # Smoothing.
        for i in range(self.smoothing):
            res = curvop(res)

        self._u = res
        return c0, c1, u


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T ** 2, 0))
    u = np.float_(phi > 0)
    return u


def add_levelset(original_levelset, new_levelset):
    result = np.array([[0 for x in range(512)] for y in range(512)])
    for i in range(512):
        for j in range(512):
            if original_levelset[i][j] == 1 or new_levelset[i][j] == 1:
                result[i][j] = 1
    return result


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


def multi_seed(macwe, seed_list, image_bw):
    u = np.array([[0 for x in range(512)] for y in range(512)])
    print("Putting seeds on coordinates:")
    for i in range(len(seed_list)):
        if u[seed_list[i][0]][seed_list[i][1]] == 0:
            print(seed_list[i][0], seed_list[i][1])
            macwe.levelset = circle_levelset(image_bw.shape, (seed_list[i][0], seed_list[i][1]), 4)
            num_iters = 0
            temp_1_c0 = 0
            temp_1_c1 = 0
            temp_2_c0 = 0
            temp_2_c1 = 0
            while True:
                # Evolve.
                c0, c1, levelset = macwe.step()
                if temp_1_c0 == c0 and temp_1_c1 == c1:
                    break
                if temp_2_c0 == c0 and temp_2_c1 == c1:
                    break
                temp_2_c0, temp_2_c1 = temp_1_c0, temp_1_c1
                temp_1_c0, temp_1_c1 = c0, c1
                num_iters += 1
            u = add_levelset(u, levelset)
    return u, num_iters


def get_seeds_from_osm(verification_pixels):
    print("Extracting seeds...")
    nbr_squares = 16
    square_size = 32
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


def write_tiff(data, output_path):
    rows = 512
    cols = 512
    # Create output tiff from data
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, 1)

    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(0, 2):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)


def error(truth_mask_array, output_mask_array):
    """ Create image with error area between output and truth masks.
    Prepares matrices for calculation of contour similarity"""

    intersection = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])
    union = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])
    intersection_nbr_elements = 0
    union_nbr_elements = 0
    result = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])
    for j in range(len(truth_mask_array)):
        for i in range(len(truth_mask_array)):
            if truth_mask_array[j][i] == 2:
                truth_mask_array[j][i] = 0
            if truth_mask_array[j][i] != output_mask_array[j][i]:
                result[j][i] = 1
            if truth_mask_array[j][i] == 0 and output_mask_array[j][i] == 1:
                intersection_nbr_elements += 1
                intersection[j][i] = 1
            if truth_mask_array[j][i] == 0 or output_mask_array[j][i] == 1:
                union_nbr_elements += 1
                union[j][i] = 1

    return jaccard_similarity(union_nbr_elements, intersection_nbr_elements)


def jaccard_similarity(union_nbr_elements, intersection_nbr_elements):
    """ Compares similarity between two contours. Returns a value between 0 and 1 (closer to 1 is more similar)."""
    similarity = intersection_nbr_elements / union_nbr_elements
    return similarity


def get_median(list):
    list.sort()
    median = int(len(list)/2)
    return list[median]


def get_mean(list):
    total = 0
    for item in list:
        total = total + item
    mean = total/len(list)
    return mean


def get_total(list):
    total = 0
    for nbr in list:
        total = total + nbr
    return total


def start_snake(img, img_nbr, validation_pixel_list, seed_list):

    # Start clock
    start = time.time()

    # Extract truth mask from validation label list
    truth_array = validation_pixel_list

    # Load image
    #img = gdal.Open(img_path)
    img_original = img.ReadAsArray()
    image_bw = rgb2gray(img_original)

    # Blur image
    gauss = gaussian_filter(image_bw, sigma=5)

    # Morphological ACWE. Initialization of the level-set.
    macwe = MorphACWE(gauss, smoothing=0, lambda1=1, lambda2=1)
    output_array, num_iters = multi_seed(macwe, seed_list, gauss)

    """Comment this line out if no output image is needed"""
    write_tiff(output_array, "output_snake/" + img_nbr + "_snake_output.tiff")

    similarity = error(truth_array, output_array)
    end = time.time()
    execution_time = end - start

    return similarity, execution_time, num_iters


files = [f for f in os.listdir("Dataset/train") if f.endswith('.shp')]
shapefiles = [os.path.join("Dataset/train", f)
              for f in files if f.endswith('.shp')]

directory = "Dataset/test_images"

all_num_iters = []
all_execution_time = []
all_similarity = []

for image in os.listdir(directory):
    if image.endswith('.tif'):
        print("Image: ", image)
        image_nbr = image.split(".")
        raster_image = gdal.Open(os.path.join(directory, image), gdal.GA_ReadOnly)
        geo_transform = raster_image.GetGeoTransform()
        projection = raster_image.GetProjectionRef()
        img_original = raster_image.ReadAsArray()
        image_bw = rgb2gray(img_original)

        verification_pixels = vectors_to_raster(shapefiles, 512, 512, geo_transform, projection)

        seed_list = get_seeds_from_osm(verification_pixels)

        if len(seed_list) > 0:
            similarity, execution_time, num_iters = start_snake(raster_image, str(image_nbr[0]), verification_pixels, seed_list)
            all_execution_time.append(execution_time)
            all_num_iters.append(num_iters)
            all_similarity.append(similarity)

        else:
            print("No seeds found")


print(" ")
print("----------------  RESULT  ------------------")
print(" ")
print("Execution time")
print("Total: ", get_total(all_execution_time))
print("Mean: ", get_mean(all_execution_time))
print("Median: ", get_median(all_execution_time))
print(" ")
print("Number of iterations")
print("Total: ", get_total(all_num_iters))
print("Mean: ", get_mean(all_num_iters))
print("Median: ", get_median(all_num_iters))
print(" ")
print("Similarity")
print("Mean: ", get_mean(all_similarity))
print("Median: ", get_median(all_similarity))