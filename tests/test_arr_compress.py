import numpy as np
import pytest
from scipy.sparse import csr_matrix

from data import SHAPE, Arr, Index
from src.geometry_analysis.arr.cast import cast_sp_to_de
from src.geometry_analysis.arr.compress import (
    compress_de_axes,
    compress_de_axis,
    compress_de_axis_to_sp,
    compress_de_coos,
    compress_de_coos_to_sp,
    compress_sp_axes,
    compress_sp_axis,
)
from src.geometry_analysis.arr.index.type_guards import is_box_slice, is_coo_loc
from utils import assert_arr_equal, assert_arr_in_place


class TestCompressDe:

    @staticmethod
    def _test_func(func, arr, idx, expected, in_place, **kwargs):

        inp = arr.copy()
        result = func(inp, idx, in_place=in_place, **kwargs)

        assert_arr_in_place(arr, inp, result, expected, in_place)
        assert_arr_equal(result, expected)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.arrs_1d + Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx", Index.a0_indices)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_axis_0(cls, arr, idx, in_place):
        expected = arr[idx]
        cls._test_func(compress_de_axis, arr, idx, expected, in_place, axis=0)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx", Index.a1_indices)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_axis_1(cls, arr, idx, in_place):
        expected = arr[:, idx]
        cls._test_func(compress_de_axis, arr, idx, expected, in_place, axis=1)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx", Index.an_indices)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_axis_none(cls, arr, idx, in_place):
        expected = arr.reshape(-1)[idx]
        cls._test_func(compress_de_axis, arr, idx, expected, in_place, axis=None)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx", Index.box_indices)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_axes(cls, arr, idx, in_place):
        expected = arr[*idx] if is_box_slice(idx) else arr[np.ix_(*idx)]
        cls._test_func(compress_de_axes, arr, idx, expected, in_place)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx", Index.coo_indices)
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_coos(cls, arr, idx, in_place):
        if is_coo_loc(idx) and len(idx[0]) == len(idx[1]) == 0:
            return
        expected = arr[idx]
        cls._test_func(compress_de_coos, arr, idx, expected, in_place)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx, c_idx", zip(Index.an_sp_indices, Index.csr_indices))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_axis_to_sp(cls, arr, idx, c_idx, in_place):
        expected = csr_matrix((arr.reshape(-1)[idx], *c_idx), shape=SHAPE)
        cls._test_func(compress_de_axis_to_sp, arr, idx, expected, in_place)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.de_arrs_2d)
    @pytest.mark.parametrize("idx, c_idx", zip(Index.coo_sp_indices, Index.csr_indices))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_de_coos_to_sp(cls, arr, idx, c_idx, in_place):
        if is_coo_loc(idx) and len(idx[0]) == len(idx[1]) == 0:
            return
        expected = csr_matrix((arr[idx], *c_idx), shape=SHAPE)
        cls._test_func(compress_de_coos_to_sp, arr, idx, expected, in_place)


class TestCompressSp:

    @staticmethod
    def _test_func(func, de_func, arr, idx, expected, reshape, in_place, **kwargs):

        inp = arr.copy()
        result = func(inp, idx, in_place=in_place, reshape=reshape, **kwargs)

        assert_arr_in_place(arr, inp, result, expected, in_place)
        assert_arr_equal(result.data, expected.data)

        if not reshape:

            if kwargs.get("axis", -1) is None:
                coo_arr = arr.tocoo()
                expected = coo_arr.data[idx]
                idx = coo_arr.row[idx] * result.shape[1] + coo_arr.col[idx]
            else:
                expected = cast_sp_to_de(expected)
            result = de_func(cast_sp_to_de(result), idx, in_place=False, **kwargs)

        assert_arr_equal(result, expected)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.sp_arrs)
    @pytest.mark.parametrize("idx", Index.a0_indices)
    @pytest.mark.parametrize("reshape", (True, False))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_sp_axis_0(cls, arr, idx, reshape, in_place):
        funcs = (compress_sp_axis, compress_de_axis)
        expected = arr[idx]
        cls._test_func(*funcs, arr, idx, expected, reshape, in_place, axis=0)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.sp_arrs)
    @pytest.mark.parametrize("idx", Index.a1_indices)
    @pytest.mark.parametrize("reshape", (True, False))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_sp_axis_1(cls, arr, idx, reshape, in_place):
        funcs = (compress_sp_axis, compress_de_axis)
        expected = arr[:, idx]
        cls._test_func(*funcs, arr, idx, expected, reshape, in_place, axis=1)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.sp_arrs)
    @pytest.mark.parametrize("idx", Index.an_indices)
    @pytest.mark.parametrize("reshape", (True, False))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_sp_axis_none(cls, arr, idx, reshape, in_place):

        funcs = (compress_sp_axis, compress_de_axis)

        def _reduce_idx_to_data_size(arr_, idx_):

            if not isinstance(idx_, slice):

                is_list = isinstance(idx_, list)
                nnz = arr_.nnz

                idx_ = np.asarray(idx_)
                idx_ = idx_[:nnz] if idx_.dtype == np.bool_ else idx_[idx_ < nnz]

                return idx_.tolist() if is_list else idx_

            return idx_

        def _apply_data_idx_to_sp(arr_, idx_):
            arr_ = arr_.tocoo()
            return csr_matrix(
                (arr_.data[idx_], (arr_.row[idx_], arr_.col[idx_])),
                dtype=arr_.dtype,
                shape=SHAPE,
            )

        idx = _reduce_idx_to_data_size(arr, idx)
        expected = _apply_data_idx_to_sp(arr, idx)

        cls._test_func(*funcs, arr, idx, expected, reshape, in_place, axis=None)

    @classmethod
    @pytest.mark.parametrize("arr", Arr.sp_arrs)
    @pytest.mark.parametrize("idx", Index.box_indices)
    @pytest.mark.parametrize("reshape", (True, False))
    @pytest.mark.parametrize("in_place", (True, False))
    def test_compress_sp_axes(cls, arr, idx, reshape, in_place):
        funcs = (compress_sp_axes, compress_de_axes)
        expected = arr[*idx] if is_box_slice(idx) else arr[np.ix_(*idx)]
        cls._test_func(*funcs, arr, idx, expected, reshape, in_place)
