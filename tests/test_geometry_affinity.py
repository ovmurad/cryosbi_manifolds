from itertools import chain

import numpy as np
import pytest

from data import EPS, Arr
from src.geometry_analysis.geometry.affinity import affinity
from utils import assert_arr_equal, assert_arr_in_place


class TestAffinity:

    dists = (Arr.x_de_dists, Arr.xy_de_dists, Arr.x_sp_dists, Arr.xy_sp_dists)
    dists = tuple(chain(*(d.values() for d in dists)))

    affs = (Arr.x_de_affs, Arr.xy_de_affs, Arr.x_sp_affs, Arr.xy_sp_affs)
    affs = tuple(chain(*(a.values() for a in affs)))

    @staticmethod
    @pytest.mark.parametrize("dists, expected", zip(dists, affs))
    @pytest.mark.parametrize("in_place", (False, True))
    @pytest.mark.parametrize("dist_is_sq", (False, True))
    def test_affinity(dists, in_place, dist_is_sq, expected):

        inp = dists.copy()

        if dist_is_sq:
            data = inp if isinstance(inp, np.ndarray) else inp.data
            data **= 2

        result = affinity(inp, EPS, dist_is_sq, in_place)
        assert_arr_in_place(dists, inp, result, expected, in_place)
        assert_arr_equal(result, expected, close=True)
