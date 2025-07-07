from typing import Iterator, Optional, Tuple, TypeAlias

from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.index import AxLoc
from .covariance import Covariance
from .eigen_decomp import eigen_decomp
from .local_function import LocalFunc

_PCA_Data: TypeAlias = Tuple[RealDeArr, RealDeArr]


class WeightedPCA(LocalFunc[_PCA_Data]):

    nouts = 2
    single_func_args = (True, False, True)
    local_func_args = (True, True)

    @staticmethod
    def single_func_raw(
        centered_x_pts: RealDeArr, weights: RealDeArr | None, ncomp: int | None
    ) -> RealDeArr:
        cov = Covariance.single_func_raw(centered_x_pts, weights)
        return eigen_decomp(cov, ncomp, is_symmetric=True)

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        weights: RealArr | None,
        ncomp: int | None,
    ) -> _PCA_Data:
        cov = Covariance.local_func_raw(x_pts, mean_pts, weights, needs_means=True)
        return eigen_decomp(cov, ncomp, is_symmetric=True)

    @classmethod
    def weighted_pca(
        cls,
        x_pts: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> _PCA_Data:

        return cls._single_func_call(
            x_pts,
            mean_pt,
            weights,
            needs_norm,
            in_place_demean,
            in_place_norm,
            ncomp=ncomp,
        )

    @classmethod
    def local_weighted_pca_iter(
        cls,
        x_pts: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[_PCA_Data]:

        return cls._local_iter_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, ncomp=ncomp
        )

    @classmethod
    def local_weighted_pca(
        cls,
        x_pts: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> _PCA_Data:

        return cls._local_batched_call(
            x_pts,
            mean_pts,
            weights,
            needs_norm,
            in_place_norm,
            bsize,
            ncomp=ncomp,
        )


weighted_pca = WeightedPCA.weighted_pca
local_weighted_pca_iter = WeightedPCA.local_weighted_pca_iter
local_weighted_pca = WeightedPCA.local_weighted_pca
