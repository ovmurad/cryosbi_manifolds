import numpy as np
import pytest

from data import NROWS, Arr
from geometry_analysis.linalg.weighted_pca import local_weighted_pca, weighted_pca
from utils import assert_arr_changed, assert_arr_equal, assert_eig_equal


class TestWeightedLocalPCA:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)
    ncomps = (None, 4)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.mean_pt_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_weighted_pca(
        x_pts,
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

        res_evals, res_evecs = weighted_pca(
            inp_x_pts,
            ncomp,
            mean_pt,
            inp_weights,
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
            exp_weights = exp_weights / np.sum(exp_weights)

        x_pts = x_pts - exp_mean_pt

        exp_weights = np.reshape(np.sqrt(exp_weights), (-1, 1))
        x_pts = x_pts * exp_weights
        _, exp_evals, exp_evecs = np.linalg.svd(x_pts, full_matrices=False)

        if ncomp is not None:
            exp_evals = exp_evals[:ncomp]
            exp_evecs = exp_evecs[:ncomp, :]

        exp_evals = exp_evals**2
        exp_evecs = exp_evecs.T

        assert_eig_equal(res_evals, res_evecs, exp_evals, exp_evecs)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize(
        "mean_pts, weights, exp_mean_pts, exp_weights", Arr.mean_pts_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_weighted_pca(
        x_pts,
        ncomp,
        mean_pts,
        weights,
        needs_norm,
        in_place_norm,
        bsize,
        exp_mean_pts,
        exp_weights,
    ):

        has_weights = weights is not None
        has_mean_pts = mean_pts is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_evals, res_evecs = local_weighted_pca(
            inp_x_pts, ncomp, mean_pts, inp_weights, needs_norm, in_place_norm, bsize
        )

        assert_arr_equal(x_pts, inp_x_pts)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if not has_mean_pts:
                exp_mean_pts = exp_mean_pts / np.sum(exp_weights, axis=1, keepdims=True)
            exp_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        for r_evals, r_evecs, exp_m_pt, exp_w in zip(
            res_evals, res_evecs, exp_mean_pts, exp_weights
        ):
            e_evals, e_evecs = weighted_pca(x_pts, ncomp, exp_m_pt, exp_w, needs_norm)
            assert_eig_equal(r_evals, r_evecs, e_evals, e_evecs)
