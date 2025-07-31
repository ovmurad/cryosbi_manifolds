from math import prod
from typing import Callable, Generic, Optional, Tuple

import numpy as np

from ..arr import DT, DeArr, SpArr
from ..create import (
    CreateSpFromDataAndSpIdx,
    create_sp_from_data_and_ax_loc,
    create_sp_from_data_and_ax_mask,
    create_sp_from_data_and_coo_loc,
    create_sp_from_data_and_coo_mask,
)
from ..index.index import (
    AxIdx,
    AxLoc,
    AxMask,
    AxSlice,
    BoxIdx,
    BoxLoc,
    BoxMask,
    BoxSlice,
    CooIdx,
    CooLoc,
    CooMask,
    SpIdx,
)
from ..index.index_len import ax_loc_len, ax_mask_len, ax_slice_len
from ..index.type_guards import (
    is_ax_loc,
    is_ax_mask,
    is_ax_slice,
    is_box_loc,
    is_box_mask,
    is_box_slice,
    is_coo_loc,
    is_coo_mask,
)


# Compress with Axis Indices


def _compress_de_with_ax_slice(
    arr: DeArr, ax_slice: AxSlice, axis: int | None, out: DeArr | None
) -> DeArr:

    if axis is None or axis == 0:
        val = arr[ax_slice]
    elif axis == 1:
        val = arr[:, ax_slice]
    else:
        raise IndexError(f"Invalid 'axis' {axis} for compressing with axis loc!")

    if out is not None:
        out[:] = val
        return out

    return val


class CompressDeWithAxIdx(Generic[AxIdx]):

    def __init__(
        self,
        compress_func: Callable[[DeArr, AxIdx, int | None, DeArr | None], DeArr],
        idx_len_func: Callable[[AxIdx, int], int],
    ):
        self.compress_func = compress_func
        self.idx_len_func = idx_len_func

    def __call__(
        self,
        arr: DeArr,
        ax_idx: AxIdx,
        axis: Optional[int] = None,
        in_place: bool = False,
    ) -> DeArr:

        if axis is None and arr.ndim == 2:
            arr = arr.reshape(-1)

        out = None

        if in_place:

            ax_len = prod(arr.shape) if axis is None else arr.shape[axis]
            idx_len = self.idx_len_func(ax_idx, ax_len)

            if axis is None or axis == 0:
                out = arr[:idx_len]
            elif axis == 1:
                out = arr[:, :idx_len]
            else:
                raise IndexError(f"Invalid 'axis' {axis}!")

        return self.compress_func(arr, ax_idx, axis, out)


compress_de_with_ax_mask = CompressDeWithAxIdx[AxMask](
    compress_func=lambda arr, ax_mask, *args: np.compress(ax_mask, arr, *args),
    idx_len_func=ax_mask_len,
)
compress_de_with_ax_loc = CompressDeWithAxIdx[AxLoc](
    compress_func=np.take,
    idx_len_func=ax_loc_len,
)
compress_de_with_ax_slice = CompressDeWithAxIdx[AxSlice](
    compress_func=_compress_de_with_ax_slice,
    idx_len_func=ax_slice_len,
)


def compress_de_axis(
    arr: DeArr, ax_idx: AxIdx, axis: Optional[int] = None, in_place: bool = False
) -> DeArr:

    if is_ax_mask(ax_idx):
        return compress_de_with_ax_mask(arr, ax_idx, axis, in_place)
    if is_ax_loc(ax_idx):
        return compress_de_with_ax_loc(arr, ax_idx, axis, in_place)
    if is_ax_slice(ax_idx):
        return compress_de_with_ax_slice(arr, ax_idx, axis, in_place)

    raise IndexError(f"Invalid 'ax_idx' {ax_idx}!")


# Compress with Box Indices


class CompressDeWithBoxIdx(Generic[BoxIdx]):

    def __init__(self, compress_de_ax_func: CompressDeWithAxIdx):
        self.compress_ax_func = compress_de_ax_func

    def __call__(self, arr: DeArr, box_idx: BoxIdx, in_place: bool = False) -> DeArr:
        arr = self.compress_ax_func(arr, box_idx[0], axis=0, in_place=in_place)
        return self.compress_ax_func(arr, box_idx[1], axis=1, in_place=in_place)


compress_de_with_box_mask = CompressDeWithBoxIdx[BoxMask](compress_de_with_ax_mask)
compress_de_with_box_loc = CompressDeWithBoxIdx[BoxLoc](compress_de_with_ax_loc)
compress_de_with_box_slice = CompressDeWithBoxIdx[BoxSlice](compress_de_with_ax_slice)


def compress_de_axes(arr: DeArr, box_idx: BoxIdx, in_place: bool = False) -> DeArr:

    if is_box_mask(box_idx):
        return compress_de_with_box_mask(arr, box_idx, in_place)
    if is_box_loc(box_idx):
        return compress_de_with_box_loc(arr, box_idx, in_place)
    if is_box_slice(box_idx):
        return compress_de_with_box_slice(arr, box_idx, in_place)

    raise IndexError(f"Invalid 'box_idx' {box_idx}!")


# Compress with Coo Indices


class CompressDeWithCooIdx(Generic[CooIdx]):

    def __init__(
        self,
        compress_ax_func: CompressDeWithAxIdx,
        ravel_func: Callable[[CooIdx, Tuple[int, ...]], AxIdx],
    ):
        self.compress_ax_func = compress_ax_func
        self.ravel_func = ravel_func

    def __call__(self, arr: DeArr, coo_idx: CooIdx, in_place: bool = False) -> DeArr:
        ravelled_idx = self.ravel_func(coo_idx, arr.shape)
        return self.compress_ax_func(arr, ravelled_idx, axis=None, in_place=in_place)


compress_de_with_coo_mask = CompressDeWithCooIdx[CooMask](
    compress_ax_func=compress_de_with_ax_mask,
    ravel_func=lambda coo_mask, _: np.asarray(coo_mask, dtype=np.bool_).reshape(-1),
)
compress_de_with_coo_loc = CompressDeWithCooIdx[CooLoc](
    compress_ax_func=compress_de_with_ax_loc,
    ravel_func=lambda coo_loc, shape: np.ravel_multi_index(coo_loc, shape),
)


def compress_de_coos(arr: DeArr, coo_idx: CooIdx, in_place: bool = False) -> DeArr:

    if is_coo_mask(coo_idx):
        return compress_de_with_coo_mask(arr, coo_idx, in_place)
    if is_coo_loc(coo_idx):
        return compress_de_with_coo_loc(arr, coo_idx, in_place)
    raise IndexError(f"Invalid 'coo_idx' {coo_idx}!")


# Compress to Sp Arr


class CompressDeToSpFunc(Generic[SpIdx]):

    def __init__(
        self,
        compress_func: CompressDeWithAxIdx | CompressDeWithCooIdx,
        create_sp_func: CreateSpFromDataAndSpIdx[SpIdx],
    ) -> None:
        self.compress_func = compress_func
        self.create_sp_func = create_sp_func

    def __call__(
        self, arr: DeArr[DT], sp_idx: SpIdx, in_place: bool = False
    ) -> SpArr[DT]:
        csr_data = self.compress_func(arr, sp_idx, in_place=in_place)
        return self.create_sp_func(csr_data, sp_idx, arr.shape)


compress_de_to_sp_with_ax_mask = CompressDeToSpFunc[AxMask](
    compress_func=compress_de_with_ax_mask,
    create_sp_func=create_sp_from_data_and_ax_mask,
)
compress_de_to_sp_with_ax_loc = CompressDeToSpFunc[AxLoc](
    compress_func=compress_de_with_ax_loc,
    create_sp_func=create_sp_from_data_and_ax_loc,
)
compress_de_to_sp_with_coo_mask = CompressDeToSpFunc[CooMask](
    compress_func=compress_de_with_coo_mask,
    create_sp_func=create_sp_from_data_and_coo_mask,
)
compress_de_to_sp_with_coo_loc = CompressDeToSpFunc[CooLoc](
    compress_func=compress_de_with_coo_loc,
    create_sp_func=create_sp_from_data_and_coo_loc,
)


# TODO: Add slice support(could be useful for diagonals)
def compress_de_axis_to_sp(
    arr: DeArr[DT], ax_idx: AxIdx, in_place: bool = False
) -> SpArr[DT]:
    if is_ax_mask(ax_idx):
        return compress_de_to_sp_with_ax_mask(arr, ax_idx, in_place)
    if is_ax_loc(ax_idx):
        return compress_de_to_sp_with_ax_loc(arr, ax_idx, in_place)
    raise IndexError(f"Invalid 'ax_idx' {ax_idx}!")


def compress_de_coos_to_sp(
    arr: DeArr[DT], coo_idx: CooIdx, in_place: bool = False
) -> SpArr[DT]:
    if is_coo_mask(coo_idx):
        return compress_de_to_sp_with_coo_mask(arr, coo_idx, in_place)
    if is_coo_loc(coo_idx):
        return compress_de_to_sp_with_coo_loc(arr, coo_idx, in_place)
    raise IndexError(f"Invalid 'coo_idx' {coo_idx}!")
