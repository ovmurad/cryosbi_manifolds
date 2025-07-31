from typing import Iterator, Optional, Tuple, TypeAlias

import numpy as np

from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.index import AxLoc
from ..linalg.covariance import Covariance
from ..linalg.eigen_decomp import eigen_decomp
from ..linalg.local_function import LocalFunc

_RMetric_Data: TypeAlias = Tuple[RealDeArr, RealDeArr]


class RMetric(LocalFunc[_RMetric_Data]):

    nouts = 2
    single_func_args = (True, False, False)
    local_func_args = (True, False)

    @staticmethod
    def _decompose_dual_metric(
        dual_metric: RealDeArr, ncomp: int | None, dual: bool
    ) -> _RMetric_Data:

        eigvals, eigvecs = eigen_decomp(dual_metric, ncomp, is_symmetric=True)
        eigvals *= 0.5

        if not dual:
            zero_eigvals_mask = eigvals == 0.0
            eigvals = np.divide(1.0, eigvals, where=~zero_eigvals_mask)
            eigvals[zero_eigvals_mask] = np.inf

        return eigvals, eigvecs

    @staticmethod
    def single_func_raw(
        centered_x_pts: RealDeArr,
        lap: RealDeArr,
        ncomp: int | None,
        dual: bool,
    ) -> _RMetric_Data:
        dual_metric = Covariance.single_func_raw(centered_x_pts, lap)
        return RMetric._decompose_dual_metric(dual_metric, ncomp, dual)

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        lap: RealArr,
        ncomp: int | None,
        dual: bool,
    ) -> _RMetric_Data:
        dual_metric = Covariance.local_func_raw(x_pts, mean_pts, lap, needs_means=False)
        return RMetric._decompose_dual_metric(dual_metric, ncomp, dual)

    @classmethod
    def rmetric(
        cls,
        x_pts: RealDeArr,
        lap: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        dual: bool = True,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> _RMetric_Data:

        return cls._single_func_call(
            x_pts,
            mean_pt,
            weights,
            needs_norm,
            in_place_demean,
            in_place_norm,
            lap=lap,
            ncomp=ncomp,
            dual=dual,
        )

    @classmethod
    def local_rmetric_iter(
        cls,
        x_pts: RealDeArr,
        lap: RealArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        dual: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[_RMetric_Data]:

        return cls._local_iter_call(
            x_pts,
            mean_pts,
            weights,
            needs_norm,
            in_place_norm,
            bsize,
            lap,
            ncomp=ncomp,
            dual=dual,
        )

    @classmethod
    def local_rmetric(
        cls,
        x_pts: RealDeArr,
        lap: RealArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        dual: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> _RMetric_Data:

        return cls._local_batched_call(
            x_pts,
            mean_pts,
            weights,
            needs_norm,
            in_place_norm,
            bsize,
            lap,
            ncomp=ncomp,
            dual=dual,
        )


rmetric = RMetric.rmetric
local_rmetric_iter = RMetric.local_rmetric_iter
local_rmetric = RMetric.local_rmetric
