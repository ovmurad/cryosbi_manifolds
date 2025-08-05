import numpy as np
import pytest

from geometry_analysis.utils.grids import create_grid_1d


class TestCreate1dGrid:

    @staticmethod
    def test_create_grid_1d_int():

        expected_result = np.array([0, 2, 4, 6])

        result = create_grid_1d(start=0, stop=6, num_steps=3, scale="int")
        assert np.allclose(result, expected_result)

        result = create_grid_1d(start=0, stop=6, step_size=2, scale="int")
        assert np.allclose(result, expected_result)

    @staticmethod
    def test_create_grid_1d_linear():

        expected_result = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

        result = create_grid_1d(start=0, stop=10, num_steps=5, scale="linear")
        assert np.allclose(result, expected_result)

        result = create_grid_1d(start=0.0, stop=10.0, step_size=2.0, scale="linear")
        assert np.allclose(result, expected_result)

    @staticmethod
    def test_create_grid_1d_exp():

        expected_result = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

        result = create_grid_1d(start=0.0, stop=4.0, num_steps=4, scale="exp", base=2.0)
        assert np.allclose(result, expected_result)

        result = create_grid_1d(
            start=0.0, stop=4.0, step_size=1.0, scale="exp", base=2.0
        )
        assert np.allclose(result, expected_result)

    @staticmethod
    def test_create_grid_1d_exceptions():

        with pytest.raises(ValueError):
            create_grid_1d(start=1.5, stop=6.7, step_size=2, scale="int")
        with pytest.raises(ValueError):
            create_grid_1d(start=100, stop=1, num_steps=2, scale="exp")
        with pytest.raises(ValueError):
            create_grid_1d(
                start=0.0, stop=10.0, step_size=1.0, num_steps=5, scale="linear"
            )
        with pytest.raises(ValueError):
            create_grid_1d(start=0.0, stop=10.0, step_size=-1.0, scale="linear")
        with pytest.raises(ValueError):
            create_grid_1d(start=0.0, stop=10.0, num_steps=0, scale="linear")
        with pytest.raises(ValueError):
            create_grid_1d(start=0, stop=6, num_steps=2, scale="unsupported")
