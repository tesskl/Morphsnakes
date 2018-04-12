# -*- coding: utf-8 -*-

from itertools import cycle
from matplotlib.image import imread
import numpy as np
from osgeo import gdal
import time
from scipy.ndimage import binary_dilation, binary_erosion
import sys

from osgeo import ogr
import subprocess
from osgeo import gdal
from osgeo import osr
from osgeo import gdalconst
from PIL import Image
import random
from osgeo import gdal, ogr
import os

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


def jaccard_similarity(union_nbr_elements, intersection_nbr_elements):
    """ Compares similarity between two contours. Returns a value between 0 and 1 (closer to 1 is more similar)."""
    similarity = intersection_nbr_elements / union_nbr_elements
    return similarity


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T ** 2, 0))
    u = np.float_(phi > 0)
    return u


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


def add_levelset(original_levelset, new_levelset):
    result = np.array([[0 for x in range(512)] for y in range(512)])
    for i in range(512):
        for j in range(512):
            if original_levelset[i][j] == 1 or new_levelset[i][j] == 1:
                result[i][j] = 1
    return result


def multi_seed_classifier(macwe, seed_list, image_bw):
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


def start_snake(img, img_nbr, validation_pixel_list, seed_list):

    # Start clock
    start = time.time()

    # Extract truth mask from validation label list
    truth_array = validation_pixel_list

    # Load image
    #img = gdal.Open(img_path)
    img_original = img.ReadAsArray()
    image_bw = rgb2gray(img_original)

    # Morphological ACWE. Initialization of the level-set.
    macwe = MorphACWE(image_bw, smoothing=0, lambda1=1, lambda2=1)
    output_array, num_iters = multi_seed_classifier(macwe, seed_list, image_bw)

    write_tiff(output_array, img_nbr + "_snake_output.tiff")
    """Comment this error line out if no truth mask is provided"""
    similarity = error(truth_array, output_array)
    end = time.time()
    execution_time = end - start

    return similarity, execution_time, num_iters
