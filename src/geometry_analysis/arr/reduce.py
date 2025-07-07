from typing import Callable, Optional

import numpy as np
from scipy.sparse import issparse

from .arr import DT, NT, Arr, IntArr, NumDeArr
from .utils import get_fill_value


def reduce_arr_with_func(
    arr: Arr,
    reduce_func: Callable[..., NumDeArr],
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> NumDeArr:

    if isinstance(arr, np.ndarray):
        return reduce_func(arr, axis=axis, keepdims=keepdims)

    arr = reduce_func(arr, axis=axis)
    if issparse(arr):
        arr = arr.data
    arr = np.asarray(arr)

    if not keepdims:
        arr = arr.item() if axis is None else arr.reshape(-1)
    elif axis is None:
        arr = arr.reshape((1, 1))
    elif axis == 0:
        arr = arr.reshape((1, -1))
    elif axis == 1:
        arr = arr.reshape((-1, 1))

    return arr


def reduce_arr_to_degrees(
    arr: Arr[DT],
    axis: Optional[int] = None,
    keepdims: bool = False,
    degree_exp: NT = 1,
) -> NumDeArr[NT]:

    degrees = reduce_arr_with_func(arr, np.sum, axis, keepdims)
    if degree_exp != 1:
        return degrees**degree_exp
    return degrees


def reduce_arr_to_nnz(
    arr: Arr[DT],
    axis: Optional[int] = None,
    keep_dims: bool = False,
    fill_value: Optional[DT] = None,
) -> IntArr:

    if isinstance(arr, np.ndarray):
        fill_value = get_fill_value(arr) if fill_value is None else fill_value
        return np.sum(fill_value != arr, axis=axis, keepdims=keep_dims)

    nnz = np.asarray(arr.getnnz(axis=axis), dtype=np.int_)

    if keep_dims:
        axis = (0, 1) if axis is None else axis
        return np.expand_dims(nnz, axis=axis)

    return nnz
