import numpy as np


def shuffle_2d_array(array):
    """Shuffle a 2D array."""
    array_copy = array.copy().ravel()
    np.random.shuffle(array_copy)
    array_copy = array_copy.reshape(array.shape)
    return array_copy


