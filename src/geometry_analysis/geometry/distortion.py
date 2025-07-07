from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

from ..arr.arr import Arr, RealArr, RealDeArr
from ..arr.compress import compress_de_axis, compress_sp_axis
from ..arr.index import AxLoc, is_ax_loc
from ..arr.threshold import threshold_arr
from ..linalg.covariance import Covariance
from ..linalg.local_function import LocalFunc
from ..linalg.tan_space_proj import TanSpaceProj
from ..linalg.weighted_pca import WeightedPCA
from ..sampling.sampling import sample_array
from .affinity import affinity
from .func_of_dist import dist
from .laplacian import laplacian


class Distortion(LocalFunc[RealDeArr]):

    nouts = 1
    single_func_args = (True, False, True)
    local_func_args = (True, True)

    @staticmethod
    def _validate_ds(ds: int | Sequence[int]) -> Tuple[int, Sequence[int]]:
        ds = (ds,) if isinstance(ds, int) else ds
        max_d = ds[-1]

        return max_d, ds

    @staticmethod
    def single_func_raw(
        centered_x_pts: RealDeArr,
        weights: RealDeArr | None,
        lap: RealDeArr,
        ds: int | Sequence[int],
    ) -> RealDeArr:

        max_d, ds = Distortion._validate_ds(ds)

        _, tan_space = WeightedPCA.single_func_raw(centered_x_pts, weights, ncomp=max_d)
        proj_pts = TanSpaceProj.single_func_raw(centered_x_pts, tan_space)

        dual_rm = Covariance.single_func_raw(proj_pts, lap)
        dual_rm *= 0.5
        dual_rm -= np.identity(max_d)

        distortions = [np.linalg.norm(dual_rm[:d, :d], ord=2) for d in ds]
        return np.array(distortions)

    @staticmethod
    def local_func_raw(
        x_pts: RealDeArr,
        mean_pts: RealDeArr,
        weights: RealArr | None,
        lap: RealArr,
        ds: int | Sequence[int],
    ) -> RealDeArr:

        max_d, ds = Distortion._validate_ds(ds)

        _, tan_spaces = WeightedPCA.local_func_raw(x_pts, mean_pts, weights, max_d)
        proj_pts = TanSpaceProj.local_func_raw(x_pts, mean_pts, lap, tan_spaces)
        if isinstance(lap, np.ndarray):
            dual_rm = np.einsum("mn,mnp,mnq->mpq", lap, proj_pts, proj_pts)
        else:
            dual_rm = np.einsum("o,op,oq->opq", lap.data, proj_pts, proj_pts)
            dual_rm = np.add.reduceat(dual_rm, indices=lap.indptr[:-1], axis=0)

        dual_rm *= 0.5
        dual_rm -= np.identity(max_d)

        distortions = [
            np.linalg.norm(dual_rm[:, :d, :d], axis=(1, 2), ord=2) for d in ds
        ]
        return np.stack(distortions, axis=1)

    @classmethod
    def distortion(
        cls,
        x_pts: RealDeArr,
        weights: RealDeArr | None,
        lap: RealDeArr,
        ds: int | Sequence[int],
        mean_pt: Optional[RealDeArr | int | bool] = None,
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
            lap=lap,
            ds=ds,
        )

    @classmethod
    def local_distortion_iter(
        cls,
        x_pts: RealDeArr,
        weights: Arr | None,
        lap: RealArr,
        ds: int | Sequence[int],
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[RealDeArr]:

        return cls._local_iter_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, lap, ds=ds
        )

    @classmethod
    def local_distortion(
        cls,
        x_pts: RealDeArr,
        weights: Arr | None,
        lap: RealArr,
        ds: int | Sequence[int],
        mean_pts: Optional[AxLoc | RealDeArr] = None,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> RealDeArr:

        return cls._local_batched_call(
            x_pts, mean_pts, weights, needs_norm, in_place_norm, bsize, lap, ds=ds
        )


distortion = Distortion.distortion
local_distortion_iter = Distortion.local_distortion_iter
local_distortion = Distortion.local_distortion


def radii_distortions(
    x_pts: RealDeArr,
    ds: int | Sequence[int],
    radii: float | Sequence[float],
    sample: AxLoc | int | float,
    rad_eps_ratio: float = 5.0,
    bsize: Optional[int] = None,
):

    radii = (radii,) if isinstance(radii, float) else radii
    radii = radii[::-1]

    if not is_ax_loc(sample):
        sample = sample_array(x_pts.shape[0], num_or_pct=sample)
    sample = np.sort(sample)
    mean_pts = x_pts[sample]

    nds = 1 if isinstance(ds, int) else len(ds)
    distortions = np.empty(shape=(nds, len(radii)), dtype=np.float_)

    dists = dist(x_pts, threshold=radii[0])

    for j, rad in enumerate(radii, start=1):

        eps = rad / rad_eps_ratio

        affs = affinity(dists, eps=eps, in_place=False)
        sample_affs = affs[sample]

        lap = laplacian(affs, eps, lap_type="geometric", in_place=True)
        if isinstance(lap, np.ndarray):
            sample_lap = compress_de_axis(lap, sample, axis=0, in_place=True)
        else:
            sample_lap = compress_sp_axis(
                lap, sample, axis=0, reshape=True, in_place=True
            )

        local_distortions = local_distortion(
            x_pts,
            sample_affs,
            sample_lap,
            ds,
            mean_pts,
            in_place_norm=True,
            needs_norm=True,
            bsize=bsize,
        )
        rad_distortion = np.mean(local_distortions, axis=0)
        print(f"Distortions for radius = {rad}: {rad_distortion}")

        distortions[:, -j] = rad_distortion

        if j < len(radii):
            dists = threshold_arr(dists, thresh=radii[j], in_place=True)

    return distortions
