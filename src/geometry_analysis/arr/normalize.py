from typing import Optional, Tuple

import numpy as np

from .arr import Arr, DeArr, RealArr, RealDeArr, RealSpArr, SpArr
from .create import create_sp_from_data_and_csr_idx
from .reduce import reduce_arr_to_degrees


def _prepare_arr_degrees_and_out(
    arr: Arr, axis: int | None, keepdims: bool, in_place: bool, degree_exp: float
) -> Tuple[RealArr, RealDeArr, RealDeArr | None]:

    if not arr.dtype == np.float_:
        arr = arr.astype(np.float_)
        in_place = True

    degrees = reduce_arr_to_degrees(arr, axis, keepdims, degree_exp)
    out = (arr if isinstance(arr, np.ndarray) else arr.data) if in_place else None

    return arr, degrees, out


def normalize_de(
    arr: DeArr,
    axis: Optional[int] = 1,
    sym_norm: bool = False,
    degree_exp: float = 1.0,
    in_place: bool = False,
) -> RealDeArr:

    if arr.ndim == 1 and (axis not in {None, 0} or sym_norm):
        raise ValueError("Invalid 'axis' or 'sym_norm' for normalizing 1d 'arr'!")

    arr, degrees, out = _prepare_arr_degrees_and_out(
        arr, axis, keepdims=True, in_place=in_place, degree_exp=degree_exp
    )

    arr = np.divide(arr, degrees, out=out)
    if sym_norm and axis is not None:
        arr = np.divide(arr, degrees.T, out=arr)

    return arr


def normalize_sp(
    arr: SpArr,
    axis: Optional[int] = 1,
    sym_norm: bool = False,
    degree_exp: float = 1.0,
    in_place: bool = False,
) -> RealSpArr:

    arr, degrees, out = _prepare_arr_degrees_and_out(
        arr, axis, keepdims=False, in_place=in_place, degree_exp=degree_exp
    )

    if axis is None:
        data = np.divide(arr.data, degrees, out=out)
    else:

        row_weights, col_weights = None, None

        if axis == 0 or sym_norm:
            col_weights = degrees[arr.indices]
        if axis == 1 or sym_norm:
            row_weights = np.repeat(degrees, np.diff(arr.indptr))

        if row_weights is not None and col_weights is not None:
            data = np.divide(arr.data, row_weights, out=out)
            data = np.divide(data, col_weights, out=out)
        else:
            weights = row_weights if row_weights is not None else col_weights
            data = np.divide(arr.data, weights, out=out)

    if in_place:
        return arr

    csr_idx = (arr.indices.copy(), arr.indptr.copy())
    return create_sp_from_data_and_csr_idx(data, csr_idx, arr.shape)


def normalize_arr(
    arr: Arr,
    axis: Optional[int] = 1,
    sym_norm: bool = False,
    degree_exp: float = 1.0,
    in_place: bool = False,
) -> RealArr:
    if isinstance(arr, np.ndarray):
        return normalize_de(arr, axis, sym_norm, degree_exp, in_place)
    return normalize_sp(arr, axis, sym_norm, degree_exp, in_place)
