from typing import Iterator, Optional

import numpy as np
from scipy.sparse import csr_matrix

from .local_function import LocalFunc
from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.index import AxLoc


class TanSpaceProj(LocalFunc[RealDeArr]):

    nouts = 1
    single_func_args = (True, False, False)
    local_func_args = (True, True)

    @staticmethod
    def single_func_raw(centered_x_pts: RealDeArr, tan_space: RealDeArr) -> RealDeArr:
        return centered_x_pts @ tan_space

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        weights: RealArr | None,
        tan_spaces: RealArr,
    ) -> RealDeArr:

        proj_mean_pts = np.einsum("mD,mDd->md", mean_pts, tan_spaces)
        proj_x_pts = np.einsum("nD,mDd->mnd", x_pts, tan_spaces)
        proj_x_pts -= np.expand_dims(proj_mean_pts, axis=1)

        if isinstance(weights, csr_matrix):
            rows = np.arange(weights.shape[0])
            rows = np.repeat(rows, repeats=np.diff(weights.indptr))

            proj_x_pts = proj_x_pts[rows, weights.indices]

        return proj_x_pts

    @classmethod
    def tan_space_proj(
        cls,
        x_pts: RealDeArr,
        tan_space: RealDeArr,
        mean_pt: Optional[RealDeArr | int | bool] = None,
        weights: Optional[RealDeArr] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> RealDeArr:

        return cls._single_func_call(
            x_pts,
            mean_pt,
            weights,
            needs_norm,
            in_place_demean,
            in_place_norm,
            tan_space=tan_space,
        )

    @classmethod
    def local_tan_space_proj_iter(
        cls,
        x_pts: RealDeArr,
        tan_spaces: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[RealDeArr]:

        return cls._local_iter_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, tan_spaces
        )

    @classmethod
    def local_tan_space_proj(
        cls,
        x_pts: RealDeArr,
        tan_spaces: RealDeArr,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> RealDeArr:

        return cls._local_batched_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, tan_spaces
        )


tan_space_proj = TanSpaceProj.tan_space_proj
local_tan_space_proj_iter = TanSpaceProj.local_tan_space_proj_iter
local_tan_space_proj = TanSpaceProj.local_tan_space_proj
