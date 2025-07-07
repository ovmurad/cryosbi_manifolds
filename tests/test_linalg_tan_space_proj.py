import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.geometry_analysis.arr.index import is_ax_loc
from src.geometry_analysis.linalg.tan_space_proj import (
    local_tan_space_proj,
    tan_space_proj,
)

from data import NROWS, Arr
from utils import assert_arr_changed, assert_arr_equal


class TestTanSpaceProj:

    bsizes = (None, 1, 3, NROWS // 4, NROWS + 100)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize("tan_space", (Arr.tan_space,))
    @pytest.mark.parametrize(
        "mean_pt, weights, exp_mean_pt, exp_weights", Arr.mean_pt_and_weights
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_demean", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    def test_tan_space_proj(
        x_pts,
        tan_space,
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

        result = tan_space_proj(
            inp_x_pts,
            tan_space,
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

        result_reproj = np.einsum("nd,Dd->nD", result, tan_space)
        result_reproj = tan_space_proj(result_reproj, tan_space)

        expected = np.einsum("nD,Dd->nd", x_pts, tan_space)

        assert_arr_equal(result, expected, close=True)
        assert_arr_equal(result_reproj, expected, close=True)

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts,))
    @pytest.mark.parametrize(
        "mean_pts, weights, tan_spaces, exp_mean_pts, exp_weights, exp_tan_spaces",
        Arr.mean_pts_weights_and_tan_spaces,
    )
    @pytest.mark.parametrize("needs_norm", (False, True))
    @pytest.mark.parametrize("in_place_norm", (False, True))
    @pytest.mark.parametrize("bsize", bsizes)
    def test_local_tan_space_proj(
        x_pts,
        tan_spaces,
        mean_pts,
        weights,
        needs_norm,
        in_place_norm,
        bsize,
        exp_mean_pts,
        exp_weights,
        exp_tan_spaces,
    ):

        has_weights = weights is not None
        has_sp_result = has_weights and isinstance(weights, csr_matrix)
        has_mean_pts = mean_pts is not None

        inp_x_pts = x_pts.copy()
        inp_weights = weights.copy() if has_weights else None

        res_proj = local_tan_space_proj(
            inp_x_pts,
            tan_spaces,
            mean_pts,
            inp_weights,
            needs_norm,
            in_place_norm,
            bsize,
        )

        assert_arr_equal(x_pts, inp_x_pts)
        if has_weights:
            assert_arr_changed(weights, inp_weights, in_place_norm and needs_norm)

        if has_sp_result:
            if is_ax_loc(mean_pts):
                weights = weights[mean_pts]
            res_proj = np.split(res_proj, indices_or_sections=weights.indptr[1:-1])

        if needs_norm:
            if not has_mean_pts:
                exp_mean_pts = exp_mean_pts / np.sum(exp_weights, axis=1, keepdims=True)
            exp_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        for r_proj, exp_ts, exp_m_pt, exp_w in zip(
            res_proj, exp_tan_spaces, exp_mean_pts, exp_weights
        ):

            e_proj = tan_space_proj(x_pts, exp_ts, exp_m_pt, exp_w, needs_norm)
            if has_sp_result:
                e_proj = e_proj[exp_w != 0]
            assert_arr_equal(r_proj, e_proj, close=True)
