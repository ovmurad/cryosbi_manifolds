from typing import Callable, Generic, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .arr import DT, DeArr, SpArr
from .index.index import AxLoc, AxMask, CooLoc, CooMask, CsrIdx, Idx, SpIdx
from .index.transforms import (
    ax_loc_to_csr_idx,
    ax_mask_to_csr_idx,
    coo_loc_to_csr_idx,
    coo_mask_to_csr_idx,
)
from .utils import get_fill_value


def create_sp_from_data_and_csr_idx(
    data: DeArr[DT] | DT, csr_idx: CsrIdx, shape: Sequence[int]
) -> SpArr[DT]:
    """
    Convoluted way of creating a new csr_matrix that ensures no data is copied.
    For some reason that seems to be the case when calling
    csr_matrix((csr_data, *csr_idx)) in some situations.

    :param data: the new sparse matrix data.
    :param csr_idx: tuple with indptr and indices.
    :param shape: shape of the new sparse matrix.

    :returns: sparse matrix that doesn't copy data.
    """

    indices = np.asarray(csr_idx[0], dtype=np.int32)
    indptr = np.asarray(csr_idx[1], dtype=np.int32)

    if not isinstance(data, np.ndarray) or data.shape != indices.shape:
        data = np.broadcast_to(data, indices.shape)

    out = csr_matrix(shape, dtype=data.dtype)

    out.data = data
    out.indices = indices
    out.indptr = indptr

    return out


class CreateSpFromDataAndSpIdx(Generic[SpIdx]):

    def __init__(
        self, sp_idx_to_csr_index_func: Callable[[SpIdx, int, int], CsrIdx]
    ) -> None:
        self.sp_idx_to_csr_index_func = sp_idx_to_csr_index_func

    def __call__(
        self, data: DeArr[DT] | DT, sp_idx: SpIdx, shape: Sequence[int]
    ) -> SpArr[DT]:
        csr_idx = self.sp_idx_to_csr_index_func(sp_idx, *shape)
        return create_sp_from_data_and_csr_idx(data, csr_idx, shape)


create_sp_from_data_and_ax_mask = CreateSpFromDataAndSpIdx[AxMask](ax_mask_to_csr_idx)
create_sp_from_data_and_ax_loc = CreateSpFromDataAndSpIdx[AxLoc](ax_loc_to_csr_idx)
create_sp_from_data_and_coo_mask = CreateSpFromDataAndSpIdx[CooMask](coo_mask_to_csr_idx)
create_sp_from_data_and_coo_loc = CreateSpFromDataAndSpIdx[CooLoc](coo_loc_to_csr_idx)


def create_de_from_data_and_idx(
    data: DeArr[DT] | DT,
    shape: Sequence[int],
    idx: Optional[Idx] = None,
    fill_value: Optional[DT] = None,
) -> DeArr[DT]:

    fill_value = get_fill_value(data) if fill_value is None else fill_value
    de_arr = np.full(fill_value=fill_value, shape=shape, dtype=data.dtype)

    if idx is not None:
        de_arr[idx] = data

    return de_arr


def create_de_from_de_and_idx(
    arr: DeArr[DT],
    idx: Idx,
    data: Optional[DT | DeArr[DT]] = None,
    in_place: bool = False,
) -> DeArr[DT]:

    if not in_place:
        arr = arr.copy()
    arr[idx] = get_fill_value(arr) if data is None else data
    return arr
