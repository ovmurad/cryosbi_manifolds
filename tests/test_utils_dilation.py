import numpy as np
import pytest
from scipy.sparse import csr_matrix

from geometry_analysis.utils.dilation import graph_dilation


class TestDilation:

    neigh = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool
    )
    vec = np.array([True, False, False, False])

    @pytest.mark.parametrize("neigh", [neigh, csr_matrix(neigh)])
    @pytest.mark.parametrize("vec", [vec])
    @pytest.mark.parametrize("n_dilations", [0, 1, 2, 3])
    def test_dilation(self, neigh, vec, n_dilations):

        expected = np.array([False, False, False, False])
        expected[: n_dilations + 1] = True

        result = graph_dilation(neigh, vec, n_dilations)

        assert np.array_equal(result, expected)
