from typing import Iterator, Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.index import AxLoc
from ..linalg.local_function import LocalFunc
from ..linalg.tan_space_proj import TanSpaceProj
from ..linalg.weighted_pca import WeightedPCA


class GradEstimation(LocalFunc[RealDeArr]):

    nouts = 1
    single_func_args = (True, False, True)
    local_func_args = (True, True)

    @staticmethod
    def single_func_raw(
        centered_x_pts: RealDeArr,
        weights: RealDeArr | None,
        f0_val: RealDeArr,
        f_vals: RealDeArr,
        ncomp: int | None,
    ) -> RealDeArr:

        _, tan_space = WeightedPCA.single_func_raw(centered_x_pts, weights, ncomp)
        proj_x_diffs = TanSpaceProj.single_func_raw(centered_x_pts, tan_space)

        f_diffs = f_vals - f0_val

        if weights is None:
            XXt = proj_x_diffs.T @ proj_x_diffs
            Xty = proj_x_diffs.T @ f_diffs
        else:
            XXt = np.einsum("mp,m,mq->pq", proj_x_diffs, weights, proj_x_diffs)
            Xty = np.einsum("mp,m,m...->p...", proj_x_diffs, weights, f_diffs)

        grad = np.linalg.solve(XXt, Xty)
        return tan_space @ grad

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        weights: RealArr | None,
        f0_vals: RealDeArr,
        f_vals: RealDeArr,
        ncomp: int | None,
    ) -> RealDeArr:

        _, tan_spaces = WeightedPCA.local_func_raw(x_pts, mean_pts, weights, ncomp)
        proj_x_diffs = TanSpaceProj.local_func_raw(x_pts, mean_pts, weights, tan_spaces)

        if isinstance(weights, np.ndarray):

            weights_data = np.expand_dims(np.sqrt(weights), axis=2)

            proj_x_diffs *= weights_data
            XXt = np.einsum("mnp,mnq->mpq", proj_x_diffs, proj_x_diffs)

            proj_x_diffs *= weights_data
            Xty = np.einsum("mnq,n...->mq...", proj_x_diffs, f_vals)

            proj_x_diffs = np.sum(proj_x_diffs, axis=1)
            Xty -= np.einsum("mq,m...->mq...", proj_x_diffs, f0_vals)

        elif isinstance(weights, csr_matrix):

            weights_data = np.expand_dims(np.sqrt(weights.data), axis=1)

            proj_x_diffs *= weights_data
            XXt = np.einsum("mp,mq->mpq", proj_x_diffs, proj_x_diffs)
            XXt = np.add.reduceat(XXt, indices=weights.indptr[:-1], axis=0)

            f_diffs = f_vals[weights.indices]
            f_diffs -= np.repeat(f0_vals, repeats=np.diff(weights.indptr), axis=0)

            proj_x_diffs *= weights_data
            Xty = np.einsum("oq,o...->oq...", proj_x_diffs, f_diffs)
            Xty = np.add.reduceat(Xty, indices=weights.indptr[:-1], axis=0)

        else:
            XXt = np.einsum("mnp,mnq->mpq", proj_x_diffs, proj_x_diffs)

            Xty = np.einsum("mnq,n...->mq...", proj_x_diffs, f_vals)

            proj_x_diffs = np.sum(proj_x_diffs, axis=1)
            Xty -= np.einsum("mq,m...->mq...", proj_x_diffs, f0_vals)

        grad = np.linalg.solve(XXt, Xty)
        return np.einsum("mDd,md...->mD...", tan_spaces, grad)

    @classmethod
    def grad_estimation(
        cls,
        x_pts: RealDeArr,
        f0_val: RealDeArr,
        f_vals: RealDeArr,
        ncomp: Optional[int] = None,
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
            f0_val=f0_val,
            f_vals=f_vals,
            ncomp=ncomp,
        )

    @classmethod
    def local_grad_estimation_iter(
        cls,
        x_pts: RealDeArr,
        f0_vals: RealDeArr,
        f_vals: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
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
            f0_vals,
            f_vals=f_vals,
            ncomp=ncomp,
        )

    @classmethod
    def local_grad_estimation(
        cls,
        x_pts: RealDeArr,
        f0_vals: RealDeArr,
        f_vals: RealDeArr,
        ncomp: Optional[int] = None,
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        weights: Optional[Arr] = None,
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
            f0_vals,
            f_vals=f_vals,
            ncomp=ncomp,
        )


grad_estimation = GradEstimation.grad_estimation
local_grad_estimation_iter = GradEstimation.local_grad_estimation_iter
local_grad_estimation = GradEstimation.local_grad_estimation
