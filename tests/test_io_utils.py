import pytest

from src.geometry_analysis.io.utils import (
    find_unique_path_in_dir,
    get_dir_paths_in_dir,
    get_file_paths_in_dir,
    get_paths_in_dir,
    is_empty_dir,
)


class TestUtils:

    dir_names = {"dir1", "dir2"}

    pkl_names = {"f1.pkl", "f2.pkl"}
    json_names = {"f1.json", "f2.json"}
    file_names = pkl_names.union(json_names)

    all_names = dir_names.union(file_names)

    extensions = {
        "": all_names,
        "pkl": pkl_names,
        "json": json_names,
        ("pkl", "json"): file_names,
        "dir": dir_names,
    }

    patterns = {
        (".*2\\..*",): {"f2.json", "f2.pkl"},
        (".*i.*",): dir_names,
        (".*1.*",): {"dir1", "f1.json", "f1.pkl"},
        (".*1", "pkl"): {"f1.pkl"},
    }

    @pytest.fixture
    def tmp_dir(self, tmp_path):

        td = tmp_path / "test_dir"
        td.mkdir()

        for name in self.dir_names:
            (td / name).mkdir()

        for name in self.file_names:
            (td / name).touch()

        return td

    @staticmethod
    @pytest.mark.parametrize("dir_name", dir_names)
    def test_is_empty_dir(tmp_dir, dir_name):
        assert is_empty_dir(tmp_dir / dir_name)
        assert not is_empty_dir(tmp_dir)

    @staticmethod
    @pytest.mark.parametrize("ext, names", extensions.items())
    def test_get_paths_in_dir_ext(tmp_dir, ext, names):

        if ext == "dir":
            result = set(p.name for p in get_dir_paths_in_dir(tmp_dir))
        else:
            result = set(p.name for p in get_paths_in_dir(tmp_dir, ext=ext))

        assert result == names

    @staticmethod
    @pytest.mark.parametrize("pattern, names", patterns.items())
    def test_get_paths_in_dir_pattern(tmp_dir, pattern, names):
        result = set(p.name for p in get_paths_in_dir(tmp_dir, *pattern))
        assert result == names

    @staticmethod
    def test_get_paths_in_dir_other_cases(tmp_dir):

        assert set(get_paths_in_dir(tmp_dir)) == set(tmp_dir.iterdir())

        result = set(get_file_paths_in_dir(tmp_dir))
        expected = set(path for path in tmp_dir.iterdir() if path.is_file())
        assert result == expected

        result = set(get_dir_paths_in_dir(tmp_dir))
        expected = set(path for path in tmp_dir.iterdir() if path.is_dir())
        assert result == expected

    @staticmethod
    def test_find_unique_path_in_dir(tmp_dir):

        with pytest.raises(IOError):
            find_unique_path_in_dir(tmp_dir, ".*1.*")
        with pytest.raises(IOError):
            find_unique_path_in_dir(tmp_dir, ext="json")
        with pytest.raises(ValueError):
            find_unique_path_in_dir(tmp_dir, ext="json", is_file=False)
        with pytest.raises(IOError):
            find_unique_path_in_dir(tmp_dir, is_file=False)
        with pytest.raises(IOError):
            find_unique_path_in_dir(tmp_dir, "f1", ["json", "pkl"])

        result = find_unique_path_in_dir(tmp_dir, "unk")
        assert result is None

        result = find_unique_path_in_dir(tmp_dir, "dir.*", is_dir=False)
        assert result is None

        result = find_unique_path_in_dir(tmp_dir, ext="npy")
        assert result is None

        result = find_unique_path_in_dir(tmp_dir, ".*1.*", is_file=False).name
        assert result == "dir1"

        result = find_unique_path_in_dir(tmp_dir, "dir1").name
        assert result == "dir1"

        result = find_unique_path_in_dir(tmp_dir, "dir1").name
        assert result == "dir1"

        result = find_unique_path_in_dir(tmp_dir, "f1", ext="json").name
        assert result == "f1.json"

        result = find_unique_path_in_dir(tmp_dir, "f1", ext="pkl", is_dir=False).name
        assert result == "f1.pkl"
