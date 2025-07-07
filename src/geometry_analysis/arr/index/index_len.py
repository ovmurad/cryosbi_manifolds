from typing import Optional

import numpy as np

from .index import AxIdx, AxLoc, AxMask, AxSlice
from .type_guards import is_ax_loc, is_ax_mask, is_ax_slice


def ax_mask_len(ax_mask: AxMask, _: Optional[int] = None) -> int:
    return np.sum(ax_mask)


def ax_loc_len(ax_loc: AxLoc, _: Optional[int] = None) -> int:
    return len(ax_loc)


def ax_slice_len(ax_slice: AxSlice, ax_len: Optional[int] = None) -> int:
    """
    Calculate the number of elements that a slice would extract.
    The slice must have `start`, `stop`, and `step` defined.
    """
    start = 0 if ax_slice.start is None else ax_slice.start
    stop = ax_len if ax_slice.stop is None else min(ax_len, ax_slice.stop)
    if stop is None:
        raise ValueError("Slice 'stop' and 'ax_len' are both None!")
    step = 1 if ax_slice.step is None else ax_slice.step
    return (stop - start + step - 1) // step


def ax_idx_len(ax_idx: AxIdx, ax_len: Optional[int] = None) -> int:
    if is_ax_mask(ax_idx):
        return ax_mask_len(ax_idx, ax_len)
    if is_ax_loc(ax_idx):
        return ax_loc_len(ax_idx, ax_len)
    if is_ax_slice(ax_idx):
        return ax_slice_len(ax_idx, ax_len)
    raise IndexError(f"Invalid 'ax_idx' {ax_idx}!")
