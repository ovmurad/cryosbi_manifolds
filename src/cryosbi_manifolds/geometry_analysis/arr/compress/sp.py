from typing import Optional, Tuple

import numpy as np

from ..arr import SpArr
from ..create import create_sp_from_data_and_csr_idx
from ..index.index import AxIdx, AxMask, BoxIdx, CsrDiffs
from ..index.transforms import ax_idx_to_ax_mask
from .de import compress_de_with_ax_mask


def _ax_idx_to_ax_mask(arr: SpArr, ax_idx: AxIdx, axis: int | None) -> AxMask:
    ax_len = arr.data.shape[0] if axis is None else arr.shape[axis]
    return np.asarray(ax_idx_to_ax_mask(ax_idx, ax_len), dtype=np.bool_)


def _update_diff_and_data_mask(
    arr: SpArr,
    row_mask: AxMask | None,
    col_mask: AxMask | None,
    data_mask: AxMask | None,
) -> Tuple[CsrDiffs, AxMask]:

    # Compute the number of col indices and data in each row.
    diff = np.diff(arr.indptr)

    # if either the cols are masked or axis is None(we are given the data_mask directly,
    # we have to recompute the diffs by summing the number of True in the data_mask).
    # if only the row_mask is given, we don't need to do this since we only need
    # to set the diffs for the removed rows to 0.
    update_diff = (col_mask is not None) or (data_mask is not None)

    if row_mask is not None:
        # expand the row mask to all the data(repeat True and False for as many cols
        # and data entries as each row has).
        data_mask = np.repeat(row_mask, diff)
        # update the diff to 0 for the rows we will remove
        diff[~row_mask] = 0

    if col_mask is not None:
        # expand the column mask to all the data or update the current data_mask
        if data_mask is None:
            data_mask = col_mask[arr.indices]
        else:
            data_mask &= col_mask[arr.indices]

    if update_diff:
        # We need to eliminate the rows that have no col indices because of the nature
        # of np.add.reduceat which computes for two repeated indices, instead of 0,
        # the value at that index.
        nnz_mask = np.flatnonzero(diff)
        nnz_indptr = arr.indptr[nnz_mask]

        # Compute the number of non-zero elements per row using the data_mask. We sum
        # the number of True's between the remaining ind ptrs and update accordingly.
        diff[nnz_mask] = np.add.reduceat(data_mask, nnz_indptr, dtype=np.int32)

    return diff, data_mask


def _update_sp(
    arr: SpArr,
    diff: CsrDiffs,
    row_mask: AxMask | None,
    col_mask: AxMask | None,
    data_mask: AxMask,
    reshape: bool,
    in_place: bool,
) -> SpArr:

    if reshape and (row_mask is not None):
        diff = compress_de_with_ax_mask(diff, row_mask)
        nrows = diff.shape[0]
    else:
        nrows = arr.shape[0]

    if in_place:
        arr.indptr = arr.indptr[: nrows + 1] if nrows < arr.shape[0] else arr.indptr
        arr.indices = compress_de_with_ax_mask(arr.indices, data_mask)
        arr.data = compress_de_with_ax_mask(arr.data, data_mask)

        indices, indptr, data = arr.indices, arr.indptr, arr.data
    else:
        indptr = np.zeros(shape=(nrows + 1,), dtype=np.int32)
        indices = arr.indices[data_mask]
        data = arr.data[data_mask]

    np.cumsum(diff, out=indptr[1:])

    if reshape and (col_mask is not None):
        keep_cols_new = np.cumsum(col_mask, dtype=np.int32)
        ncols = keep_cols_new[-1]
        np.take(keep_cols_new - 1, indices=indices, out=indices)
    else:
        ncols = arr.shape[1]

    if in_place:
        arr._shape = (nrows, ncols)
        return arr

    shape = (nrows, ncols)
    csr_idx = (indices, indptr)
    return create_sp_from_data_and_csr_idx(data, csr_idx, shape)


# Compress Sparse with Ax Index


def compress_sp_axis(
    arr: SpArr,
    ax_idx: AxIdx,
    axis: Optional[int] = None,
    reshape: bool = False,
    in_place: bool = False,
) -> SpArr:

    ax_mask = _ax_idx_to_ax_mask(arr, ax_idx, axis)

    row_mask = ax_mask if axis == 0 else None
    col_mask = ax_mask if axis == 1 else None
    data_mask = ax_mask if axis is None else None

    diff, data_mask = _update_diff_and_data_mask(arr, row_mask, col_mask, data_mask)
    return _update_sp(arr, diff, row_mask, col_mask, data_mask, reshape, in_place)


# Compress Sparse with Box Index


def compress_sp_axes(
    arr: SpArr, box_idx: BoxIdx, reshape: bool = False, in_place: bool = False
) -> SpArr:

    row_mask = _ax_idx_to_ax_mask(arr, box_idx[0], axis=0)
    col_mask = _ax_idx_to_ax_mask(arr, box_idx[1], axis=1)
    data_mask = None

    diff, data_mask = _update_diff_and_data_mask(arr, row_mask, col_mask, data_mask)
    return _update_sp(arr, diff, row_mask, col_mask, data_mask, reshape, in_place)


# TODO: Implement more as needed
