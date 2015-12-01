"""This module contains methods that can aid in preprocessing a pixel stream
"""
import numpy as np
from scipy.misc import imresize

# from util.listops import product

# _module_mem = dict()


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


def scale(array, wanted_shape=(64,64)):
    """Scales an array down to the wanted shape.
    """
    return imresize(array.squeeze(), wanted_shape, 'bilinear'))

