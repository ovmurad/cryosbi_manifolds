from itertools import chain

import numpy as np
import pytest

from data import EPS, Arr
from src.geometry_analysis.geometry.laplacian import laplacian
from utils import assert_arr_equal, assert_arr_in_place


class TestLaplacian:

    lap_types = {"random_walk", "geometric", "symmetric"}
    lap_affs = tuple(chain(Arr.de_lap_affs.items(), Arr.sp_lap_affs.items()))

    @staticmethod
    @pytest.mark.parametrize("r, affs", lap_affs)
    @pytest.mark.parametrize("eps", (None, EPS))
    @pytest.mark.parametrize("lap_type", lap_types)
    @pytest.mark.parametrize("diag_add", (1.0, 2.0))
    @pytest.mark.parametrize("aff_minus_id", (False, True))
    @pytest.mark.parametrize("in_place", (False, True))
    def test_laplacian(affs, r, eps, lap_type, diag_add, aff_minus_id, in_place):

        inp = affs.copy()
        result = laplacian(inp, eps, lap_type, diag_add, aff_minus_id, in_place)

        expected = Arr.de_laps[r] if isinstance(affs, np.ndarray) else Arr.sp_laps[r]
        expected = expected[lap_type].copy()

        if aff_minus_id:
            diag_add = -(diag_add - 1.0)
        else:
            expected = -expected
            diag_add = diag_add - 1.0

        expected[np.diag_indices(expected.shape[0])] += diag_add

        if eps is not None:
            data = expected if isinstance(affs, np.ndarray) else expected.data
            data *= 4.0 / (EPS**2)

        assert_arr_in_place(affs, inp, result, expected, in_place)
        assert_arr_equal(result, expected, close=True)
