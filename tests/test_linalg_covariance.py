import numpy as np
import pytest

from data import NROWS, Arr
from src.geometry_analysis.linalg.covariance import covariance, local_covariance
from utils import assert_arr_changed, assert_arr_equal


class TestCovariance:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.mean_pt_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_covariance(
        x_pts,
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

        result = covariance(
            inp_x_pts, mean_pt, inp_weights, needs_norm, in_place_demean, in_place_norm
        )

        assert_arr_changed(x_pts, inp_x_pts, in_place_demean and mean_pt is not None)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if mean_pt is True:
                exp_mean_pt = exp_mean_pt / np.sum(exp_weights)
            exp_weights = exp_weights / np.sum(exp_weights)

        x_pts = x_pts - exp_mean_pt

        if mean_pt is True and (needs_norm or not has_weights):
            expected = np.cov(x_pts, rowvar=False, ddof=0, aweights=exp_weights)
        else:
            x_pts = x_pts * np.reshape(np.sqrt(exp_weights), (-1, 1))
            expected = x_pts.T @ x_pts

        assert_arr_equal(result, expected, close=True)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pts, weights, exp_mean_pts, exp_weights", Arr.mean_pts_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_covariance(
        x_pts,
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

        res_covs = local_covariance(
            inp_x_pts,
            mean_pts,
            inp_weights,
            needs_norm=needs_norm,
            in_place_norm=in_place_norm,
            bsize=bsize,
        )

        assert_arr_equal(x_pts, inp_x_pts)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if needs_norm:
            if not has_mean_pts:
                exp_mean_pts = exp_mean_pts / np.sum(exp_weights, axis=1, keepdims=True)
            exp_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        for res_cov, exp_m_pt, exp_w in zip(res_covs, exp_mean_pts, exp_weights):
            exp_cov = covariance(x_pts, exp_m_pt, exp_w, needs_norm)
            assert_arr_equal(res_cov, exp_cov, close=True)
