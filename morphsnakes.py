# -*- coding: utf-8 -*-

from itertools import cycle
from matplotlib.image import imread
import numpy as np
from osgeo import gdal
import time
from scipy.ndimage import binary_dilation, binary_erosion

# COLORS = ["#008000", "#003cb3"]
COLORS = ["#ffffff", "#000000"]


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

border_pixels = []


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
        return c0, c1

    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in range(iterations):
            self.step()


''' Compares similarity between two contours. Returns a value between 0 and 1 (closer to 1 is more similar).
'''


def jaccard_similarity(union_nbr_elements, intersection_nbr_elements):
    similarity = intersection_nbr_elements / union_nbr_elements
    print("---------------------------")
    print("Similarity: ", similarity, " %")


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T ** 2, 0))
    u = np.float_(phi > 0)
    return u

def two_circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid2 = np.mgrid[list(map(slice, shape))].T - (10, 10)
    phi2 = 10 - np.sqrt(np.sum(grid2.T ** 2, 0))
    u2 = np.float_(phi2 > 0)

    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T ** 2, 0))
    u = np.float_(phi > 0)

    total = u + u2
    return total


def write_tiff(data, output_path, geo_transform, projection, rows, cols):
    # Create output tiff from data
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, 1)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
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


def error(truth_path, output_path, geo_transform, projection, rows, cols, directory_path):
    """ Create image with error area between output and truth masks.
    Prepares matrices for calculation of contour similarity"""

    truth_mask = gdal.Open(truth_path)
    output_mask = gdal.Open(output_path)
    truth_mask_array = np.array(truth_mask.ReadAsArray())
    output_mask_array = np.array(output_mask.ReadAsArray())

    intersection = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])
    union = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])
    intersection_nbr_elements = 0
    union_nbr_elements = 0
    result = np.array([[0 for x in range(len(truth_mask_array))] for y in range(len(truth_mask_array))])

    for j in range(len(truth_mask_array)):
        for i in range(len(truth_mask_array)):
            if truth_mask_array[j][i] == 255:
                truth_mask_array[j][i] = 1
            if truth_mask_array[j][i] != output_mask_array[j][i]:
                result[j][i] = 1
            if truth_mask_array[j][i] == 0 and output_mask_array[j][i] == 1:
                intersection_nbr_elements += 1
                intersection[j][i] = 1
            if truth_mask_array[j][i] == 0 or output_mask_array[j][i] == 1:
                union_nbr_elements += 1
                union[j][i] = 1
    write_tiff(intersection, directory_path +"/intersection.tiff", geo_transform, projection, rows, cols)
    write_tiff(union, directory_path + "/union.tiff", geo_transform, projection, rows, cols)
    write_tiff(result, directory_path + "/error.tiff", geo_transform, projection, rows, cols)
    print("Calculating similarity...")

    jaccard_similarity(union_nbr_elements, intersection_nbr_elements)


def start_snake():
    # Load original image
    directory_path = "two_side_water"
    img_path = directory_path + "/two_side_water_small.tif"
    img_original = imread(img_path)
    image_bw = rgb2gray(img_original)

    # Path to truth mask
    truth_path = directory_path + "/truth_mask.tif"

    # Define path to output
    output_path = directory_path + "/output.tiff"

    # Save projection, geo transform and shape from original image
    img_data = gdal.Open(img_path)
    geo_transform = img_data.GetGeoTransform()
    projection = img_data.GetProjectionRef()
    rows, cols = img_original.shape[0], img_original.shape[1]

    # Morphological ACWE. Initialization of the level-set.
    print("Running algorithm...")
    macwe = MorphACWE(image_bw, smoothing=1, lambda1=10000, lambda2=10000)

    """Use one or two circles depending on what imaged is used (one or two water masses)"""
    #macwe.levelset = circle_levelset(image_bw.shape, (image_bw.shape[0] / 2, image_bw.shape[1] / 2), 50)
    macwe.levelset = two_circle_levelset(image_bw.shape, (image_bw.shape[0] / 2, image_bw.shape[1] / 2), 50)

    num_iters = 0
    temp_1_c0 = 0
    temp_1_c1 = 0
    temp_2_c0 = 0
    temp_2_c1 = 0
    while True:
        # Evolve.
        c0, c1 = macwe.step()
        if temp_1_c0 == c0 and temp_1_c1 == c1:
            break
        if temp_2_c0 == c0 and temp_2_c1 == c1:
            break
        temp_2_c0, temp_2_c1 = temp_1_c0, temp_1_c1
        temp_1_c0, temp_1_c1 = c0, c1
        num_iters += 1

    # Create output and error files
    print("Creating output files...")
    write_tiff(macwe.levelset, output_path, geo_transform, projection, rows, cols)

    """Comment this error line out if no truth mask is provided"""
    #error(truth_path, output_path, geo_transform, projection, rows, cols, directory_path)

    return num_iters


if __name__ == '__main__':
    start = time.time()
    num_iters = start_snake()
    end = time.time()

    print("Number of iterations required: ", num_iters)
    print("Execution time: ", end - start, " s")
