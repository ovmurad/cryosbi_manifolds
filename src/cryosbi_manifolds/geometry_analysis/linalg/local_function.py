from itertools import compress, cycle
from typing import Any, Callable, Generic, Iterator, Optional, Tuple, TypeVar

import numpy as np

from ..arr.arr import Arr, RealDeArr
from ..arr.batch import iter_arr_batches, iter_de_batches
from ..arr.index import AxLoc, is_ax_loc
from ..arr.normalize import normalize_arr

_LocalData = TypeVar("_LocalData", RealDeArr, Tuple[RealDeArr, RealDeArr])


class LocalFunc(Generic[_LocalData]):

    nouts: int = 1

    single_func_args: Tuple[bool, bool, bool] = (True, True, True)
    local_func_args: Tuple[bool, bool] = (True, True)

    single_func_raw: Callable[..., _LocalData] = None
    local_func_raw: Callable[..., _LocalData] = None

    @staticmethod
    def prepare_x_pts_mean_pt_and_weights(
        x_pts: RealDeArr,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> Tuple[RealDeArr, RealDeArr | None, RealDeArr | None]:

        if needs_norm and weights is not None:
            weights = normalize_arr(weights, axis=None, in_place=in_place_norm)

        if mean_pt is True:
            if weights is None:
                mean_pt = np.mean(x_pts, axis=0)
            else:
                mean_pt = np.sum(x_pts * np.expand_dims(weights, axis=1), axis=0)
        elif isinstance(mean_pt, int):
            mean_pt = x_pts[mean_pt]

        if mean_pt is not None:
            x_pts = np.subtract(x_pts, mean_pt, out=x_pts if in_place_demean else None)

        return x_pts, mean_pt, weights

    @classmethod
    def prepare_local_mean_pts_and_arrs(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        *arrs: Arr,
    ) -> Tuple[Tuple[RealDeArr, Arr | None], Tuple[Arr, ...]]:

        has_weights = weights is not None

        if needs_norm and has_weights:
            weights = normalize_arr(weights, axis=1, in_place=in_place_norm)

        if is_ax_loc(mean_pts):
            weights = weights[mean_pts] if has_weights else None
            arrs = tuple(arr[mean_pts] for arr in arrs)
            mean_pts = x_pts[mean_pts]

        if mean_pts is None:
            mean_pts = x_pts if weights is None else weights @ x_pts

        return (mean_pts, weights), arrs

    @classmethod
    def _single_func_call(
        cls,
        x_pts: RealDeArr,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
        **kwargs: Any,
    ) -> _LocalData:

        x_pts_mean_pt_and_weights = cls.prepare_x_pts_mean_pt_and_weights(
            x_pts, mean_pt, weights, needs_norm, in_place_demean, in_place_norm
        )

        args = compress(x_pts_mean_pt_and_weights, cls.single_func_args)
        return cls.single_func_raw(*args, **kwargs)

    @classmethod
    def _local_func_call(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        *arrs: Arr,
        **kwargs: Any,
    ) -> _LocalData:

        mean_pts_and_weights, arrs = cls.prepare_local_mean_pts_and_arrs(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, *arrs
        )

        mean_pts_and_weights = compress(mean_pts_and_weights, cls.local_func_args)
        return cls.local_func_raw(x_pts, *mean_pts_and_weights, *arrs, **kwargs)

    @classmethod
    def _local_iter_call(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
        *arrs: Arr,
        **kwargs: Any,
    ) -> Iterator[_LocalData]:

        pre_format = in_place_norm and needs_norm

        if pre_format:
            (mean_pts, weights), arrs = cls.prepare_local_mean_pts_and_arrs(
                x_pts, mean_pts, weights, needs_norm, in_place_norm, *arrs
            )
            needs_norm = False

        if bsize is None:
            mean_pts_and_weights, arrs = ((mean_pts, weights), ), (arrs, )
        else:
            if weights is None and mean_pts is None:
                mean_pts = np.arange(x_pts.shape[0])

            has_mean_locs = is_ax_loc(mean_pts)

            if weights is None or has_mean_locs:
                weights = cycle((weights,))
            else:
                weights = iter_arr_batches(weights, bsize)

            if not len(arrs) or has_mean_locs:
                arrs = cycle((arrs,))
            else:
                arrs = zip(*(iter_arr_batches(arr, bsize) for arr in arrs))

            if mean_pts is None:
                mean_pts = cycle((None,))
            else:
                mean_pts = iter_de_batches(mean_pts, bsize)

            mean_pts_and_weights = zip(mean_pts, weights)

        if pre_format:
            return (
                cls.local_func_raw(x_pts, *compress(mp_w, cls.local_func_args), *a, **kwargs)
                for mp_w, a in zip(mean_pts_and_weights, arrs)
            )

        return (
            cls._local_func_call(x_pts, *mp_w, needs_norm, False, *a, **kwargs)
            for mp_w, a in zip(mean_pts_and_weights, arrs)
        )

    @classmethod
    def _local_batched_call(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
        *arrs: Arr,
        **kwargs: Any,
    ) -> _LocalData:

        if bsize is None:
            return cls._local_func_call(
                x_pts, mean_pts, weights, needs_norm, in_place_norm, *arrs, **kwargs
            )

        local_data_iter = cls._local_iter_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, *arrs, **kwargs
        )

        if cls.nouts == 1:
            return np.concatenate(list(local_data_iter))

        local_data_batches = [[] for _ in range(cls.nouts)]
        for ld in local_data_iter:
            for out_ld_b, out in zip(local_data_batches, ld):
                out_ld_b.append(out)
        return tuple(np.concatenate(ld_batches) for ld_batches in local_data_batches)
