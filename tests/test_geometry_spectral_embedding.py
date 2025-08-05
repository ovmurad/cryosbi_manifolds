from itertools import chain

import numpy as np
import pytest
from megaman.embedding.spectral_embedding import (
    spectral_embedding as megaman_spectral_embedding,
)
from megaman.geometry import Geometry
from geometry_analysis.geometry.spectral_embedding import spectral_embedding
from geometry_analysis.sampling.constants import RANDOM_STATE

from data import EPS, Arr
from utils import assert_arr_changed, assert_eig_equal


class TestSpectralEmbedding:

    affs = chain(Arr.sp_lap_affs.values(), Arr.de_lap_affs.values())
    lap_type_to_megaman_lap_type = {
        "random_walk": "randomwalk",
        "geometric": "geometric",
        "symmetric": "symmetricnormalized",
    }
    lap_types = ("random_walk", "geometric", "symmetric")
    eigen_solvers = ("dense", "arpack", "lobpcg", "amg")

    @classmethod
    @pytest.mark.parametrize("affs", affs)
    @pytest.mark.parametrize("ncomp", (2, 4))
    @pytest.mark.parametrize("lap_type", lap_types)
    @pytest.mark.parametrize("eigen_solver", eigen_solvers)
    @pytest.mark.parametrize("in_place", (False, True))
    def test_spectral_embedding(cls, affs, ncomp, lap_type, eigen_solver, in_place):

        if not isinstance(affs, np.ndarray) and eigen_solver == "dense":
            return

        geom = Geometry(
            laplacian_method=cls.lap_type_to_megaman_lap_type[lap_type],
            laplacian_kwds={"scaling_epps": EPS, "symmetrize_input": False},
        )
        geom.affinity_matrix = affs.copy()
        _, exp_eigvals, exp_eigvecs = megaman_spectral_embedding(
            geom, ncomp, eigen_solver, RANDOM_STATE
        )
        exp_eigvals = np.real(exp_eigvals)
        exp_eigvals *= -1.0

        if eigen_solver in {"amg", "lobpcg"}:
            if lap_type == "symmetric":
                exp_eigvals -= 1.0
            else:
                exp_eigvals *= 4.0 / (EPS**2)

        inp = affs.copy()
        res_eigvals, res_eigvecs = spectral_embedding(
            inp, ncomp, EPS, lap_type, eigen_solver, in_place=in_place
        )

        if not in_place:
            assert_arr_changed(affs, inp, in_place)

        if (eigen_solver == "dense" and (lap_type in {"random_walk", "geometric"})) or (
            eigen_solver == "arpack" and lap_type == "symmetric"
        ):
            return

        assert_eig_equal(
            res_eigvals,
            res_eigvecs,
            exp_eigvals,
            exp_eigvecs,
            check_spectrum="smallest",
            is_symmetric=(lap_type == "symmetric"),
        )
