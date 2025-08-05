import numpy as np
import pytest
from scipy.sparse import csr_matrix

from geometry_analysis.io.file_io import (
    delete_any,
    io_handler_factory,
    load_any,
    save_any,
)


class TestFileIO:

    test_data = {
        "npy": (np.array([1, 2, 3]), np.random.rand(3, 3)),
        "npz": (csr_matrix(np.eye(3)), csr_matrix((3, 5), dtype=np.int_)),
        "json": ({"a": 1, "b": 2}, {"a": 1, "b": {"c": 1, "d": 2}}),
        "pkl": ({"a": 1, "b": 2}, {0: np.array([1, 2, 3]), 1: csr_matrix(np.eye(3))}),
    }

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        td = tmp_path / "test_dir"
        td.mkdir()
        return td

    @staticmethod
    def _test_equal(datum, loaded_datum, ext):
        match ext:
            case "npy":
                np.testing.assert_array_equal(datum, loaded_datum)
            case "npz":
                np.testing.assert_array_equal(datum.todense(), loaded_datum.todense())
            case "json":
                assert isinstance(datum, dict)
                assert datum == loaded_datum
            case "pkl":
                assert isinstance(datum, dict)
                assert set(datum.keys()) == set(loaded_datum.keys())
                for k in datum.keys():
                    expected = datum[k]
                    result = loaded_datum[k]
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(expected, result)
                    elif isinstance(expected, csr_matrix):
                        np.testing.assert_array_equal(
                            expected.todense(), result.todense()
                        )
                    else:
                        assert datum == loaded_datum
            case _:
                raise ValueError

    @staticmethod
    @pytest.mark.parametrize("ext, data", zip(test_data.keys(), test_data.values()))
    def test_save_and_load_file_exceptions(tmp_dir, ext, data):

        file_path = tmp_dir / "test"

        with pytest.raises(IsADirectoryError):
            save_any(tmp_dir, data[0])

        with pytest.raises(NotImplementedError):
            save_any(file_path, data[0], "jpeg")

        with pytest.raises(NotImplementedError):
            load_any(file_path, "jpeg")

        with pytest.raises(IOError):
            load_any(file_path)

        with pytest.raises(FileNotFoundError):
            load_any(tmp_dir / "test_1")

        with pytest.raises(FileNotFoundError):
            load_any(tmp_dir / "test_1", ext)

        with pytest.raises(IOError):
            load_any(file_path.with_suffix(".npy"), "json")

    @staticmethod
    @pytest.mark.parametrize("ext, data", zip(test_data.keys(), test_data.values()))
    def test_save_and_load_file(tmp_dir, ext, data):

        inferred_ext = "pkl" if ext == "json" else ext

        io_handler = io_handler_factory(ext)
        assert io_handler.ext == ext

        for datum in data:

            file_path = tmp_dir / "test"

            save_any(file_path, datum, ext)
            loaded_datum = load_any(file_path, ext)
            TestFileIO._test_equal(datum, loaded_datum, ext)

            io_handler.save(file_path, datum)
            loaded_datum = io_handler.load(file_path)
            TestFileIO._test_equal(datum, loaded_datum, ext)

            delete_any(file_path, ext)
            assert not file_path.with_suffix(f".{ext}").exists()

            file_path = tmp_dir / "test_1"
            save_any(file_path, datum)
            loaded_datum = load_any(file_path)
            TestFileIO._test_equal(datum, loaded_datum, inferred_ext)

            file_path = (tmp_dir / "test_2").with_suffix(f".{ext}")
            save_any(file_path, datum)
            loaded_datum = load_any(file_path)
            TestFileIO._test_equal(datum, loaded_datum, ext)

            io_handler.delete(file_path)
            assert not file_path.with_suffix(f".{ext}").exists()
