import numpy as np

from .arr import DT, Arr, DeArr


def _diagonal_slice(ncols: int) -> slice:
    return slice(None, None, ncols + 1)


def set_arr_diag(arr: Arr[DT], a: DT | DeArr[DT]) -> Arr[DT]:

    if isinstance(arr, np.ndarray):
        arr.reshape(-1)[_diagonal_slice(arr.shape[1])] = a
    else:
        arr.setdiag(a)

    return arr


def add_to_arr_diag(arr: Arr[DT], a: DT | DeArr[DT]) -> Arr[DT]:

    if isinstance(arr, np.ndarray):
        arr.reshape(-1)[_diagonal_slice(arr.shape[1])] += a
    else:
        arr.setdiag(arr.diagonal() + a)

    return arr
