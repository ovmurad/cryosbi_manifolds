import numpy as np
import pytest
from scipy.sparse import csr_matrix

from data import NON_TRIVIAL_THRESHOLDS, THRESHOLDS, Arr
from geometry_analysis.geometry import count_x, count_xy, count_y, dist, neigh
from src.geometry_analysis.arr.cast import cast_sp_to_de
from utils import assert_arr_changed, assert_arr_equal


class TestFuncOfDist:

    funcs = (dist, neigh, count_x, count_y, count_xy)
    expected_funcs = (
        lambda x: x,
        lambda x: x != np.inf,
        lambda x: np.sum(x != np.inf, axis=1),
        lambda x: np.sum(x != np.inf, axis=0),
        (lambda x: np.sum(x != np.inf, axis=1), lambda x: np.sum(x != np.inf, axis=0)),
    )

    thresholds = (None,) + THRESHOLDS + (NON_TRIVIAL_THRESHOLDS,)
    working_memories = (None, 0.01, 10.0)
    pts = ((Arr.x_pts, None), (Arr.x_pts, Arr.y_pts))

    @staticmethod
    @pytest.mark.parametrize("func, expected_func", zip(funcs, expected_funcs))
    @pytest.mark.parametrize("x_pts, y_pts", pts)
    @pytest.mark.parametrize("threshold", thresholds)
    @pytest.mark.parametrize("return_sp", (False, True))
    @pytest.mark.parametrize("sp_dists", (None, False, True))
    @pytest.mark.parametrize("wm", working_memories)
    def test_distance_mat(
        func, expected_func, x_pts, y_pts, threshold, return_sp, sp_dists, wm
    ):

        has_y_pts = y_pts is not None
        has_mult_funcs = isinstance(expected_func, tuple)
        has_mult_thresh = isinstance(threshold, tuple)

        dists, dists_inp = None, None

        if sp_dists is not None:

            th = max(threshold) if has_mult_thresh else threshold
            if has_y_pts:
                dists = Arr.xy_sp_dists[th] if sp_dists else Arr.xy_de_dists[th]
            else:
                dists = Arr.x_sp_dists[th] if sp_dists else Arr.x_de_dists[th]

            x_pts = y_pts = None
            dists_inp = dists.copy()

        result = func(x_pts, y_pts, dists_inp, threshold, return_sp, working_memory=wm)

        if dists is not None:
            if has_mult_thresh:
                assert_arr_changed(dists, dists_inp, should_change=True)
            elif isinstance(dists, csr_matrix):
                assert_arr_equal(dists, dists_inp)

        if has_mult_funcs:
            assert len(result) == len(expected_func)
            if has_mult_thresh:
                result = [
                    [th_to_res[th] for th in threshold] for th_to_res in result.values()
                ]
            else:
                result = [[r] for r in result]
        else:
            if has_mult_thresh:
                result = [[result[th] for th in threshold]]
            else:
                result = [[result]]

            expected_func = [
                expected_func,
            ]

        if not has_mult_thresh:
            threshold = [threshold]

        for exp_func, result_func in zip(expected_func, result):
            for th, result_th in zip(threshold, result_func):

                if func in {dist, neigh}:
                    if (sp_dists or return_sp) and th is not None:
                        assert isinstance(result_th, csr_matrix)
                        result_th = cast_sp_to_de(result_th)
                    else:
                        assert isinstance(result_th, np.ndarray)
                else:
                    assert isinstance(result_th, np.ndarray)

                expected = Arr.xy_de_dists[th] if has_y_pts else Arr.x_de_dists[th]
                expected = exp_func(expected)

                assert_arr_equal(result_th, expected, close=True)
