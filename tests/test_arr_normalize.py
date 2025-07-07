import numpy as np
import pytest
from scipy.sparse import csr_matrix

from data import Arr
from src.geometry_analysis.arr.cast import cast_sp_to_de
from src.geometry_analysis.arr.normalize import normalize_arr
from utils import assert_arr_equal, assert_arr_in_place, mask_de


class TestNormalize:

    arrs = Arr.arrs_1d + Arr.de_arrs_2d + Arr.sp_non_empty_arrs

    @staticmethod
    @pytest.mark.parametrize("arr", arrs)
    @pytest.mark.parametrize("sym_norm", (True, False))
    @pytest.mark.parametrize("axis", (0, 1, None))
    @pytest.mark.parametrize("degree_exp", (0.5, 1.0))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_normalize(arr, axis, sym_norm, degree_exp, in_place):

        if arr.ndim == 1 and (sym_norm or axis == 1):
            with pytest.raises(ValueError):
                inp = arr.copy()
                normalize_arr(inp, axis, sym_norm, degree_exp, in_place)
            return

        if sym_norm:
            min_size = min(arr.shape)
            arr = arr[:min_size, :min_size].copy()

        inp = arr.copy()
        result = normalize_arr(inp, axis, sym_norm, degree_exp, in_place)

        if isinstance(inp, csr_matrix):
            expected = cast_sp_to_de(arr, fill_value=0)
        else:
            expected = arr.copy()
        expected = expected.astype(np.float_)

        weights = np.sum(expected, axis=axis, keepdims=True) ** degree_exp
        expected /= weights
        if sym_norm and axis is not None:
            expected /= weights.T

        if isinstance(arr, csr_matrix):
            expected = mask_de(expected, rm_value=0, return_sp=True)

        in_place &= arr.dtype == np.float_

        assert_arr_in_place(arr, inp, result, expected, in_place)
        assert_arr_equal(result, expected, close=True, fill_value=0)
