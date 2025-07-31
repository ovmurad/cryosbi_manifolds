import numpy as np

from .index import AxIdx, AxLoc, AxMask, CooLoc, CooMask, CsrIdx
from .type_guards import is_ax_mask


def ax_idx_to_ax_mask(ax_idx: AxIdx, ax_len: int) -> AxMask:

    if is_ax_mask(ax_idx):
        return ax_idx

    ax_mask = np.full(ax_len, fill_value=False, dtype=np.bool_)
    ax_mask[ax_idx] = True
    return ax_mask


def ax_mask_to_csr_idx(ax_mask: AxMask, nrows: int, ncols: int) -> CsrIdx:
    return ax_loc_to_csr_idx(np.flatnonzero(ax_mask), nrows, ncols)


def ax_loc_to_csr_idx(ax_loc: AxLoc, nrows: int, ncols: int) -> CsrIdx:

    ax_loc = np.asarray(ax_loc, dtype=np.int32)

    indptr = np.arange(0, nrows * ncols + 1, ncols, dtype=np.int32)
    indptr = np.asarray(np.searchsorted(ax_loc, indptr), dtype=np.int32)

    return ax_loc % ncols, indptr


def coo_mask_to_csr_idx(coo_mask: CooMask, nrows: int, ncols: int) -> CsrIdx:

    indptr = np.zeros(nrows + 1, dtype=np.int32)
    np.cumsum(np.count_nonzero(coo_mask, axis=1), out=indptr[1:])

    indices = np.asarray(np.flatnonzero(coo_mask), dtype=np.int32)
    indices %= ncols

    return indices, indptr


def coo_loc_to_csr_idx(coo_loc: CooLoc, nrows: int, _: int) -> CsrIdx:

    indptr = np.arange(nrows + 1, dtype=np.int32)
    indptr = np.asarray(np.searchsorted(coo_loc[0], indptr), dtype=np.int32)

    return np.asarray(coo_loc[1], dtype=np.int32), indptr
