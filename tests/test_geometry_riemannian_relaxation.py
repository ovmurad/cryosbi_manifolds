import numpy as np
import pytest

from data import SUBSPACE_DIM, Arr
from src.geometry_analysis.geometry.affinity import affinity
from src.geometry_analysis.geometry.func_of_dist import dist
from src.geometry_analysis.geometry.laplacian import laplacian
from src.geometry_analysis.geometry.riemannian_relaxation import riemannian_relaxation
from src.geometry_analysis.geometry.rmetric import local_rmetric


class TestRiemannianRelaxation:

    @staticmethod
    @pytest.mark.parametrize("x_pts", (Arr.x_pts_subspace,))
    def test_riemannian_relaxation(x_pts):

        dists = dist(x_pts, threshold=0.8)
        affs = affinity(dists, eps=0.2)
        lap = laplacian(affs, eps=0.2, lap_type="geometric")

        x_pts += np.random.normal(size=x_pts.shape, scale=0.01)
        x_pts *= 3.0

        print(local_rmetric(x_pts, lap)[0].mean(axis=0))
        new_x_pts = riemannian_relaxation(
            emb_pts=x_pts,
            lap=lap,
            lap_eps=0.2,
            d=SUBSPACE_DIM,
            orth_eps=0.1,
            maxiter=100,
        )
        print(local_rmetric(new_x_pts, lap)[0].mean(axis=0))
