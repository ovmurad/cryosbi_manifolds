from itertools import cycle

import numpy as np
import pytest

from data import NROWS, SHAPE, Index
from geometry_analysis.arr.index.transforms import (
    ax_idx_to_ax_mask,
    ax_loc_to_csr_idx,
    ax_mask_to_csr_idx,
    coo_loc_to_csr_idx,
    coo_mask_to_csr_idx,
)
from geometry_analysis.arr.index.type_guards import (
    is_ax_loc,
    is_ax_mask,
    is_ax_slice,
    is_box_loc,
    is_box_mask,
    is_box_slice,
    is_coo_loc,
    is_coo_mask,
    is_csr_idx,
)


class TestTypeGuards:

    type_guards = {
        is_ax_mask: Index.a0_masks + Index.a1_masks + Index.an_masks,
        is_ax_loc: Index.a0_locs + Index.a1_locs + Index.an_locs,
        is_ax_slice: Index.a0_slices + Index.a1_slices + Index.an_slices,
        is_box_mask: Index.box_masks,
        is_box_loc: Index.box_locs,
        is_box_slice: Index.box_slices,
        is_coo_mask: Index.coo_masks,
        is_coo_loc: Index.coo_locs,
    }

    @classmethod
    @pytest.mark.parametrize("type_guard, indices", type_guards.items())
    def test_type_guards(cls, type_guard, indices):

        for index in indices:

            assert type_guard(index)

            for other_type_guard in cls.type_guards.keys():

                if type_guard == other_type_guard:
                    continue
                if {type_guard, other_type_guard} == {is_coo_loc, is_box_loc}:
                    continue

                assert (
                    not other_type_guard(index)
                    or (len(index) == 0)
                    or (len(index) == 2 and len(index[0]) == len(index[1]) == 0)
                )


class TestTransformations:

    ax_mask_to_csr_idx_params = tuple(zip(Index.an_masks, Index.csr_indices))
    ax_loc_to_csr_idx_params = tuple(zip(Index.an_locs, Index.csr_indices))
    coo_mask_to_csr_idx_params = tuple(zip(Index.coo_masks, Index.csr_indices))
    coo_loc_to_csr_idx_params = tuple(zip(Index.coo_locs, Index.csr_indices))
    ax_idx_to_ax_mask_params = tuple(zip(Index.a0_indices, cycle(Index.a0_masks)))

    @staticmethod
    def _test_arr_index_to_csr_idx(func, idx, expected):

        result = func(idx, *SHAPE)

        assert is_csr_idx(result)

        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])

    @classmethod
    @pytest.mark.parametrize("ax_mask, expected", ax_mask_to_csr_idx_params)
    def test_ax_mask_to_csr_idx(cls, ax_mask, expected):
        cls._test_arr_index_to_csr_idx(ax_mask_to_csr_idx, ax_mask, expected)

    @classmethod
    @pytest.mark.parametrize("ax_loc, expected", ax_loc_to_csr_idx_params)
    def test_ax_loc_to_csr_idx(cls, ax_loc, expected):
        cls._test_arr_index_to_csr_idx(ax_loc_to_csr_idx, ax_loc, expected)

    @classmethod
    @pytest.mark.parametrize("coo_mask, expected", coo_mask_to_csr_idx_params)
    def test_coo_mask_to_csr_idx(cls, coo_mask, expected):
        cls._test_arr_index_to_csr_idx(coo_mask_to_csr_idx, coo_mask, expected)

    @classmethod
    @pytest.mark.parametrize("coo_loc, expected", coo_loc_to_csr_idx_params)
    def test_coo_loc_to_csr_idx(cls, coo_loc, expected):
        cls._test_arr_index_to_csr_idx(coo_loc_to_csr_idx, coo_loc, expected)

    @staticmethod
    @pytest.mark.parametrize("ax_idx, expected", ax_idx_to_ax_mask_params)
    def test_ax_idx_to_ax_mask(ax_idx, expected):

        result = ax_idx_to_ax_mask(ax_idx, NROWS)

        assert is_ax_mask(result)
        np.testing.assert_array_equal(result, expected)
