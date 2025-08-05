import pytest

from data import Arr
from geometry_analysis.arr.threshold import threshold_arr
from utils import assert_arr_equal, assert_arr_in_place, mask_de, mask_sp


class TestThresholding:

    thresholds = (0.5, 5.0)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.num_de_arrs_2d + Arr.num_arrs_1d)
    @pytest.mark.parametrize("thresh", thresholds)
    @pytest.mark.parametrize("in_place", (True, False))
    @pytest.mark.parametrize("return_sp", (True, False))
    def test_threshold_de(arr, thresh, in_place, return_sp):

        inp = arr.copy()
        result = threshold_arr(inp, thresh, return_sp, in_place)

        return_sp = return_sp and arr.ndim == 2
        expected = mask_de(arr.copy(), thresh=thresh, return_sp=return_sp)

        assert_arr_equal(result, expected)
        assert_arr_in_place(arr, inp, result, expected, in_place)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.num_sp_arrs)
    @pytest.mark.parametrize("thresh", thresholds)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_threshold_sp(arr, thresh, in_place):

        inp = arr.copy()
        result = threshold_arr(inp, thresh, in_place=in_place)

        expected = mask_sp(arr, thresh=thresh)

        assert_arr_equal(result, expected)
        assert_arr_in_place(arr, inp, result, expected, in_place)
