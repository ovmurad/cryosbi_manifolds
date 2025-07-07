import numpy as np


def get_fill_value(arr):

    dtype = arr.dtype

    if dtype == np.bool_:
        return False
    elif dtype == np.int_:
        return np.iinfo(np.int_).max
    elif dtype == np.float_:
        return np.inf

    raise ValueError(f"Unknown 'dtype': {dtype}!")
