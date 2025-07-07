import numpy as np
import pytest
from megaman.geometry.affinity import compute_affinity_matrix
from megaman.geometry.laplacian import compute_laplacian_matrix
from megaman.utils.estimate_radius import compute_nbr_wts
from megaman.utils.estimate_radius import distortion as megaman_distortion

from data import NROWS, Arr
from src.geometry_analysis.geometry.distortion import (
    distortion,
    local_distortion,
    radii_distortions,
)
from src.geometry_analysis.geometry.func_of_dist import dist
from src.geometry_analysis.sampling.sampling import sample_array
from utils import assert_arr_changed, assert_arr_equal


class TestDistortion:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)
    ds = (3, (1, 2, 3, 4))

    radii = (2.0, 3.0, 4.0)
    rad_eps_ratio = 3.0
    sample = sample_array(NROWS, num_or_pct=0.5)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.mean_pt_and_weights
    )
    @pytest.mark.parametrize("lap", (Arr.geometric_laps[0][0],))
    @pytest.mark.parametrize("ds", ds)
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_distortion(
        x_pts,
        weights,
        lap,
        ds,
        mean_pt,
        needs_norm,
        in_place_demean,
        in_place_norm,
        exp_mean_pt,
        exp_weights,
    ):

        has_weights = weights is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        result = distortion(
            inp_x_pts,
            inp_weights,
            lap,
            ds,
            mean_pt,
            needs_norm,
            in_place_demean,
            in_place_norm,
        )

        assert_arr_changed(x_pts, inp_x_pts, in_place_demean and mean_pt is not None)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if mean_pt is True:
                exp_mean_pt = exp_mean_pt / np.sum(exp_weights)

        x_pts = x_pts - exp_mean_pt

        exp_weights = np.reshape(np.sqrt(exp_weights), (-1, 1))
        _, _, exp_evecs = np.linalg.svd(x_pts * exp_weights, full_matrices=False)

        if isinstance(ds, int):
            ds = (ds,)

        for d, res in zip(ds, result):

            exp_proj = np.einsum("nD,dD->nd", x_pts, exp_evecs[:d, :])

            cov = np.einsum("np,nq->npq", exp_proj, exp_proj)
            cov *= np.expand_dims(lap, axis=(1, 2))
            exp_dual_rm = 0.5 * np.sum(cov, axis=0)

            _, exp_evals, _ = np.linalg.svd(exp_dual_rm - np.identity(d))
            exp = np.max(np.abs(exp_evals))

            assert_arr_equal(res, exp, close=True)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pts, weights, lap, exp_mean_pts, exp_weights, exp_lap",
        Arr.mean_pts_weights_and_laps,
    )
    @pytest.mark.parametrize("ds", ds)
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_distortion(
        x_pts,
        weights,
        lap,
        ds,
        mean_pts,
        needs_norm,
        in_place_norm,
        bsize,
        exp_mean_pts,
        exp_weights,
        exp_lap,
    ):

        has_weights = weights is not None
        has_mean_pts = mean_pts is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_distortion = local_distortion(
            inp_x_pts,
            inp_weights,
            lap,
            ds,
            mean_pts,
            needs_norm,
            in_place_norm,
            bsize,
        )

        assert_arr_equal(x_pts, inp_x_pts)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if not has_mean_pts:
                exp_mean_pts = exp_mean_pts / np.sum(exp_weights, axis=1, keepdims=True)
            exp_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        for r_distortion, exp_l, exp_m_pt, exp_w in zip(
            res_distortion, exp_lap, exp_mean_pts, exp_weights
        ):
            e_distortion = distortion(x_pts, exp_w, exp_l, ds, exp_m_pt, needs_norm)
            assert_arr_equal(r_distortion, e_distortion, close=True)

    @classmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    def test_radii_distortion(cls, x_pts):

        results = radii_distortions(
            x_pts, cls.ds[1], cls.radii, cls.sample, cls.rad_eps_ratio
        )

        for i, d in enumerate(cls.ds[1]):
            for j, rad in enumerate(cls.radii):

                dists = dist(x_pts, threshold=rad, return_sp=True)
                h = rad / cls.rad_eps_ratio
                A = compute_affinity_matrix(dists, "gaussian", radius=h)

                (PS, nbrs) = compute_nbr_wts(A, cls.sample)
                L = compute_laplacian_matrix(A, method="geometric", scaling_epps=h)
                L = L.tocsr()
                e_dist = megaman_distortion(x_pts, L, cls.sample, PS, nbrs, NROWS, d)

                assert np.abs(e_dist - results[i, j]) < 0.03
