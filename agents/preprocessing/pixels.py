"""This module contains methods that can aid in preprocessing a pixel stream
"""
import numpy as np
from util import product

_module_mem = dict()


def pixel_max(*frames):
    """Takes any number of frames as separate arguments and
    returns a pixel buffer with the maximum value per pixel across
    the frames given.
    """
    return np.max(np.stack(frames), axis=0)


def pixel_avg(*frames):
    """Takes any number of frames as separate arguments and
    returns a pixel buffer with the average value per pixel across
    the frames given.
    """
    return np.average(np.stack(frames), axis=0)


def scale(array2d, wanted_shape):
    """Scales a 2d array (one frame) down to the wanted size.
    It is expected that wanted_shape are smaller than array2d.shape.
    """
    raise NotImplementedError(
        "Need to figure out something like bilinear interpolation")
