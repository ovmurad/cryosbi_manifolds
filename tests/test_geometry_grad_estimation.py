import numpy as np
import pytest

from data import NROWS, SUBSPACE_DIM, Arr
from src.geometry_analysis.geometry.grad_estimation import (
    grad_estimation,
    local_grad_estimation,
)
from utils import assert_arr_changed, assert_arr_equal


class TestGradEstimation:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)
    ncomps = (SUBSPACE_DIM // 2, SUBSPACE_DIM)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts_subspace,))
    @pytest.mark.parametrize(
        "f0_val, f_vals",
        ((Arr.f1_vals[0], Arr.f1_vals), (Arr.f_vals[0], Arr.f_vals)),
    )
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.subsp_mean_pt_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_grad_estimation(
        x_pts,
        f0_val,
        f_vals,
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

        result = grad_estimation(
            inp_x_pts,
            f0_val,
            f_vals,
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

        x_pts = x_pts - exp_mean_pt

        exp_weights = np.reshape(np.sqrt(exp_weights), (-1, 1))
        _, _, exp_evecs = np.linalg.svd(x_pts * exp_weights, full_matrices=False)
        exp_evecs = exp_evecs[:ncomp, :]

        exp_proj = np.einsum("nD,dD->nd", x_pts, exp_evecs)
        exp_proj = exp_proj * exp_weights

        if f_vals.ndim == 1:
            exp_diffs = (f_vals - f0_val) * exp_weights.flatten()
            expected = np.linalg.lstsq(exp_proj, exp_diffs)[0]
        else:
            exp_diffs = (f_vals - f0_val) * exp_weights
            expected = np.stack(
                [np.linalg.lstsq(exp_proj, ed)[0] for ed in exp_diffs.T], axis=0
            )

        expected = np.einsum("...d,dD->D...", expected, exp_evecs)
        assert_arr_equal(result, expected, rtol=1e-3, atol=1e-5)

        if mean_pt is not None and mean_pt is not True:
            proj_result = np.einsum("D...,Dd->...d", result, Arr.tan_space)
            proj_result = np.einsum("...d,Dd->D...", proj_result, Arr.tan_space)
            assert_arr_equal(result, proj_result, rtol=1e-1, atol=1e-1)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts_subspace,))
    @pytest.mark.parametrize(
        "mean_pts, weights, f0_vals, f_vals, exp_mean_pts, exp_weights, exp_f0_vals",
        Arr.subsp_mean_pts_weights_and_f_vals,
    )
    @pytest.mark.parametrize("ncomp", ncomps)
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_grad_estimation(
        x_pts,
        f0_vals,
        f_vals,
        ncomp,
        mean_pts,
        weights,
        needs_norm,
        in_place_norm,
        bsize,
        exp_mean_pts,
        exp_weights,
        exp_f0_vals,
    ):

        has_weights = weights is not None
        has_mean_pts = mean_pts is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_grad = local_grad_estimation(
            inp_x_pts,
            f0_vals,
            f_vals,
            ncomp,
            mean_pts,
            inp_weights,
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

        for r_grad, e_f0_val, exp_m_pt, exp_w in zip(
            res_grad, exp_f0_vals, exp_mean_pts, exp_weights
        ):
            e_grad = grad_estimation(
                x_pts, e_f0_val, f_vals, ncomp, exp_m_pt, exp_w, needs_norm
            )
            assert_arr_equal(r_grad, e_grad, close=True)
