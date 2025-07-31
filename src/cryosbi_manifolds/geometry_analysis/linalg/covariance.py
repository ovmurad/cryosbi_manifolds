from typing import Iterator, Optional

import numpy as np

from .local_function import LocalFunc
from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.index import AxLoc
from ..arr.reduce import reduce_arr_to_degrees


class Covariance(LocalFunc[RealDeArr]):

    nouts = 1
    single_func_args = (True, False, True)
    local_func_args = (True, True)

    @staticmethod
    def single_func_raw(
        centered_x_pts: RealDeArr, weights: RealDeArr | None
    ) -> RealDeArr:

        if weights is None:
            return (1.0 / centered_x_pts.shape[0]) * (centered_x_pts.T @ centered_x_pts)
        return np.einsum("mp,m,mq->pq", centered_x_pts, weights, centered_x_pts)

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        weights: RealArr | None,
        needs_means: bool,
    ) -> RealDeArr:

        nmeans = mean_pts.shape[0] if weights is None else weights.shape[0]

        if weights is None:
            cov = (1.0 / x_pts.shape[0]) * (x_pts.T @ x_pts)
            cov = np.repeat(np.expand_dims(cov, axis=0), repeats=nmeans, axis=0)
        elif isinstance(weights, np.ndarray):
            cov = np.einsum("mn,np,nq->mpq", weights, x_pts, x_pts)
        else:
            x_pts_iter = np.split(x_pts[weights.indices], weights.indptr[1:-1])
            weights_iter = np.split(weights.data, weights.indptr[1:-1])

            cov = np.stack([(xp.T * w) @ xp for xp, w in zip(x_pts_iter, weights_iter)])

        if weights is None:
            means_outer_x_pts = np.einsum("mp,q->mpq", mean_pts, np.mean(x_pts, axis=0))
        else:
            means_outer_x_pts = np.einsum("mp,mq->mpq", mean_pts, weights @ x_pts)

        cov -= means_outer_x_pts
        cov -= np.transpose(means_outer_x_pts, axes=(0, 2, 1))

        if needs_means:

            if weights is None:
                outer_means = np.einsum("mp,mq->mpq", mean_pts, mean_pts)
            else:
                weights = reduce_arr_to_degrees(weights, axis=1)
                outer_means = np.einsum("mp,m,mq->mpq", mean_pts, weights, mean_pts)

            cov += outer_means

        return cov

    @classmethod
    def covariance(
        cls,
        x_pts: RealDeArr,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> RealDeArr:

        return cls._single_func_call(
            x_pts, mean_pt, weights, needs_norm, in_place_demean, in_place_norm
        )

    @classmethod
    def local_covariance_iter(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_means: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[RealDeArr]:

        return cls._local_iter_call(
            x_pts,
            mean_pts,
            weights,
            needs_norm,
            in_place_norm,
            bsize,
            needs_means=needs_means,
        )

    @classmethod
    def local_covariance(
        cls,
        x_pts: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_means: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> RealDeArr:

        return cls._local_batched_call(
            x_pts,
            mean_pts,
            weights,
            needs_norm,
            in_place_norm,
            bsize,
            needs_means=needs_means,
        )


covariance = Covariance.covariance
local_covariance_iter = Covariance.local_covariance_iter
local_covariance = Covariance.local_covariance
