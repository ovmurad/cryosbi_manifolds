import numpy as np
import pytest

from data import NROWS, Arr
from geometry_analysis.arr.batch import iter_arr_batches
from utils import assert_arr_equal, assert_arr_in_place


class TestBatching:

    bsizes = (1, 3, NROWS // 4, NROWS + 100)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs)
    @pytest.mark.parametrize("bsize", bsizes)
    def test_batch_dense(arr, bsize):

        result = iter_arr_batches(arr, bsize)
        expected = np.split(arr, np.arange(bsize, NROWS, bsize))

        for b_result, b_expected in zip(result, expected):
            assert_arr_equal(b_result, b_expected)
            assert_arr_in_place(arr, arr, b_result, b_expected, in_place=bsize > NROWS)

    @staticmethod
    @pytest.mark.parametrize("arr", Arr.sp_arrs)
    @pytest.mark.parametrize("bsize", bsizes)
    def test_batch_sparse(arr, bsize):

        result = iter_arr_batches(arr, bsize)

        b_start_indices = np.arange(0, NROWS, bsize)
        b_stop_indices = np.arange(bsize, NROWS + bsize, bsize)

        expected = (
            arr[start:stop] for start, stop in zip(b_start_indices, b_stop_indices)
        )

        for b_result, b_expected in zip(result, expected):
            assert_arr_equal(b_result, b_expected)
            assert_arr_in_place(arr, arr, b_result, b_expected, in_place=bsize > NROWS)
