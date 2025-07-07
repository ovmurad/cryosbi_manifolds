from typing import Any, Callable, Type, TypeGuard

import numpy as np
from numpy.typing import NDArray

from .index import (
    AxLoc,
    AxMask,
    AxSlice,
    BoxLoc,
    BoxMask,
    BoxSlice,
    CooLoc,
    CooMask,
    CsrIdx,
    Idx,
)


def _is_arr_of_type(idx: Idx, item_dtype: Type[np.generic], ndim: int) -> bool:
    return (
        isinstance(idx, np.ndarray)
        and np.issubdtype(idx.dtype, item_dtype)
        and idx.ndim == ndim
    )


def _is_seq_of_type(
    idx: Idx,
    seq_type: Type[tuple] | Type[list] | Type[NDArray],
    item_type: Type[bool] | Type[int] | Callable[[Any], TypeGuard[Idx]],
    ndim: int | None,
) -> bool:

    if not isinstance(idx, seq_type):
        return False

    if ndim is not None:
        if len(idx) != ndim:
            return False

    if len(idx) == 0:
        return True
    elif isinstance(item_type, type):
        return type(idx[0]) is item_type
    return item_type(idx[0])


# Ax Index Type Guards


def is_ax_mask(idx: Idx) -> TypeGuard[AxMask]:
    if _is_arr_of_type(idx, item_dtype=np.bool_, ndim=1):
        return idx.size > 0
    if _is_seq_of_type(idx, seq_type=list, item_type=bool, ndim=None):
        return len(idx) > 0
    return False


def is_ax_loc(idx: Idx) -> TypeGuard[AxLoc]:
    if _is_arr_of_type(idx, item_dtype=np.integer, ndim=1):
        return True
    if _is_seq_of_type(idx, seq_type=list, item_type=int, ndim=None):
        return True
    return False


def is_ax_slice(idx: Idx) -> TypeGuard[AxSlice]:
    return isinstance(idx, slice)


# Box Index Type Guards


def is_box_mask(idx: Idx) -> TypeGuard[BoxMask]:
    return _is_seq_of_type(idx, seq_type=tuple, item_type=is_ax_mask, ndim=2)


def is_box_loc(idx: Idx) -> TypeGuard[BoxLoc]:
    return _is_seq_of_type(idx, seq_type=tuple, item_type=is_ax_loc, ndim=2)


def is_box_slice(idx: Idx) -> TypeGuard[BoxSlice]:
    return _is_seq_of_type(idx, seq_type=tuple, item_type=is_ax_slice, ndim=2)


# Coo Index Type Guards


def is_coo_mask(idx: Idx) -> TypeGuard[CooMask]:
    if _is_arr_of_type(idx, item_dtype=np.bool_, ndim=2):
        return True
    if _is_seq_of_type(idx, seq_type=list, item_type=is_ax_mask, ndim=None):
        return True
    return False


def is_coo_loc(idx: Idx) -> TypeGuard[CooLoc]:
    return _is_seq_of_type(idx, seq_type=tuple, item_type=is_ax_loc, ndim=2)


# Csr Index Type Guards


def is_csr_idx(idx: Idx) -> TypeGuard[CsrIdx]:
    if _is_seq_of_type(idx, seq_type=tuple, item_type=np.ndarray, ndim=2):
        return True
    if _is_arr_of_type(idx[0], item_dtype=np.int32, ndim=1):
        return True
    return False
