import numpy as np
import pytest

from data import Arr
from src.geometry_analysis.arr.reduce import (
    reduce_arr_to_degrees,
    reduce_arr_to_nnz,
    reduce_arr_with_func,
)
from utils import assert_arr_equal, cast_to_de


class TestReduce:

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d + Arr.sp_arrs)
    @pytest.mark.parametrize("reduce_func", (np.sum, np.mean))
    @pytest.mark.parametrize("axis", (0, 1, None))
    @pytest.mark.parametrize("keep_dims", (False, True))
    def test_reduce_arr_with_func(arr, reduce_func, axis, keep_dims):

        result = reduce_arr_with_func(arr, reduce_func, axis, keep_dims)

        arr = cast_to_de(arr, fill_value=0)
        expected = reduce_func(arr, axis=axis, keepdims=keep_dims)

        assert_arr_equal(result, expected, close=True)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d + Arr.sp_arrs)
    @pytest.mark.parametrize("axis", (0, 1, None))
    @pytest.mark.parametrize("keep_dims", (False, True))
    @pytest.mark.parametrize("degree_exp", (0.5, 1.0))
    def test_reduce_arr_to_degrees(arr, axis, keep_dims, degree_exp):

        result = reduce_arr_to_degrees(arr, axis, keep_dims, degree_exp)

        arr = cast_to_de(arr, fill_value=0)
        expected = np.sum(arr, axis=axis, keepdims=keep_dims)

        if degree_exp != 1:
            expected = expected**degree_exp

        assert_arr_equal(result, expected, close=True)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.num_de_arrs_2d + Arr.num_sp_arrs)
    @pytest.mark.parametrize("axis", (0, 1, None))
    @pytest.mark.parametrize("keep_dims", (False, True))
    @pytest.mark.parametrize("fill_value", (None, -1))
    def test_reduce_arr_to_nnz(arr, axis, keep_dims, fill_value):

        result = reduce_arr_to_nnz(arr, axis, keep_dims, fill_value)

        arr = cast_to_de(arr, fill_value=-1)
        expected = np.sum(arr != -1, axis=axis, keepdims=keep_dims)

        assert_arr_equal(result, expected)
