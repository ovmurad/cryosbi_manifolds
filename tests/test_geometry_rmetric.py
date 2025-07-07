import numpy as np
import pytest
from megaman.geometry import RiemannMetric
from src.geometry_analysis.geometry.rmetric import local_rmetric, rmetric

from data import NROWS, Arr
from utils import assert_arr_changed, assert_arr_equal, assert_eig_equal


class TestRMetric:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)
    ncomps = (None, 4)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize("lap", (Arr.geometric_laps[0][0],))
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.mean_pt_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_rmetric(
        x_pts,
        lap,
        ncomp,
        mean_pt,
        weights,
        needs_norm,
        in_place_demean,
        in_place_norm,
        exp_mean_pt,
        exp_weights,
    ):

        has_weights = weights is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_dual_evals, res_dual_evecs = rmetric(
            inp_x_pts,
            lap,
            ncomp,
            mean_pt,
            inp_weights,
            True,
            needs_norm,
            in_place_demean,
            in_place_norm,
        )

        assert_arr_changed(x_pts, inp_x_pts, in_place_demean and mean_pt is not None)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_evals, res_evecs = rmetric(
            inp_x_pts,
            lap,
            ncomp,
            mean_pt,
            inp_weights,
            False,
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
        cov = np.einsum("np,nq->npq", x_pts, x_pts)
        cov *= np.expand_dims(lap, axis=(1, 2))
        cov = 0.5 * np.sum(cov, axis=0)

        exp_dual_evals, exp_dual_evecs = np.linalg.eigh(cov)
        exp_dual_evals, exp_dual_evecs = exp_dual_evals[::-1], exp_dual_evecs[:, ::-1]

        if ncomp is not None:
            exp_dual_evals = exp_dual_evals[:ncomp]
            exp_dual_evecs = exp_dual_evecs[:, :ncomp]

        assert_eig_equal(res_dual_evals, res_dual_evecs, exp_dual_evals, exp_dual_evecs)
        assert_eig_equal(1.0 / res_dual_evals, res_evecs, res_evals, res_evecs)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pts, weights, lap, exp_mean_pts, exp_weights, exp_lap",
        Arr.mean_pts_weights_and_laps,
    )
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_rmetric(
        x_pts,
        lap,
        ncomp,
        mean_pts,
        weights,
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
        inp_lap = lap.copy()
        inp_weights = weights.copy() if has_weights else None

        res_evals, res_evecs = local_rmetric(
            inp_x_pts,
            inp_lap,
            ncomp,
            mean_pts,
            inp_weights,
            True,
            needs_norm,
            in_place_norm,
            bsize,
        )

        assert_arr_equal(x_pts, inp_x_pts)
        assert_arr_equal(lap, inp_lap)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if not has_mean_pts:
                exp_mean_pts = exp_mean_pts / np.sum(exp_weights, axis=1, keepdims=True)
            exp_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        for r_evals, r_evecs, exp_l, exp_m_pt, exp_w in zip(
            res_evals, res_evecs, exp_lap, exp_mean_pts, exp_weights
        ):
            e_evals, e_evecs = rmetric(
                x_pts, exp_l, ncomp, exp_m_pt, exp_w, dual=True, needs_norm=needs_norm
            )
            assert_eig_equal(r_evals, r_evecs, e_evals, e_evecs)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize("lap", Arr.geometric_laps)
    @pytest.mark.parametrize("ncomp", (3, 5))
    def test_rmetric_megaman(x_pts, lap, ncomp):

        megaman_rmetric = RiemannMetric(x_pts, lap)
        megaman_rmetric.get_dual_rmetric()

        res_eigvals, res_eigvecs = local_rmetric(x_pts, lap, ncomp)
        exp_eigvals = megaman_rmetric.Hsvals[:, :ncomp]
        exp_eigvecs = megaman_rmetric.Hvv[:, :ncomp].transpose(0, 2, 1)

        assert_eig_equal(
            res_eigvals,
            res_eigvecs,
            exp_eigvals,
            exp_eigvecs,
            check_spectrum="largest",
            is_symmetric=True,
        )

        res_eigvals, res_eigvecs = local_rmetric(x_pts, lap, ncomp, dual=False)
        exp_eigvals = megaman_rmetric.Gsvals[:, :ncomp]
        exp_eigvecs = megaman_rmetric.Hvv[:, :ncomp].transpose(0, 2, 1)

        assert_eig_equal(
            res_eigvals,
            res_eigvecs,
            exp_eigvals,
            exp_eigvecs,
            check_spectrum="smallest",
            is_symmetric=True,
        )
