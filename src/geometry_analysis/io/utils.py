import re
from pathlib import Path
from typing import Iterator, Sequence


def get_path_ext(path: Path) -> str:
    return path.suffix[1:]


def _glob_with_regex(dir_path: Path, pattern: str) -> Iterator[Path]:
    """Filter the paths in a dir using regex. Return an iterator paths found."""
    re_pattern = re.compile(pattern)
    return filter(lambda path: re_pattern.match(path.name), dir_path.iterdir())


def is_empty_dir(dir_path: Path) -> bool:
    return next(dir_path.iterdir(), None) is None


def get_paths_in_dir(
    dir_path: Path, pattern: str = ".*", ext: str | Sequence[str] = ""
) -> Iterator[Path]:
    """
    Get all paths in a given folder. Optionally filter paths by pattern and/or
    extension. The paths we're looking for will have the form {pattern}.{ext}.
    This function uses regex and not glob style patterns.

    :param dir_path: the folder to get the paths from, either a string or a Path object.
    :param pattern: The pattern that we're trying to match.
    :param ext: string of the file extension to filter by or a sequence of such strings.

    :return: An iterator yielding Path objects for each path in the folder which match
    the provided pattern and extensions.
    """
    if ext:
        ext = [ext] if isinstance(ext, str) else ext
        pattern = f"{pattern}\\.({"|".join(ext)})"

    return _glob_with_regex(dir_path, pattern)


def get_file_paths_in_dir(
    dir_path: Path, pattern: str = ".*", ext: str | Sequence[str] = ""
) -> Iterator[Path]:
    """
    Get all file paths in a given folder. Optionally filter files by pattern and/or
    extension. The paths we're looking for will have the form {pattern}.{ext}.
    This function uses regex and not glob style patterns.

    :param dir_path: the folder to get the paths from, either a string or a Path object.
    :param pattern: The pattern that we're trying to match.
    :param ext: string of the file extension to filter by or a sequence of such strings.

    :return: An iterator yielding Path objects for each file in the folder which matches
    the provided pattern and extensions.
    """
    return filter(lambda path: path.is_file(), get_paths_in_dir(dir_path, pattern, ext))


def get_dir_paths_in_dir(dir_path: Path, pattern: str = ".*") -> Iterator[Path]:
    """
    Get all dir paths in a given folder. Optionally filter folders by pattern.
    This function uses regex and not glob style patterns.

    :param dir_path: the folder to get the paths from, either a string or a Path object.
    :param pattern: The pattern that we're trying to match.

    :return: An iterator yielding Path objects for each folder that matches the provided
    pattern.
    """
    return filter(lambda path: path.is_dir(), get_paths_in_dir(dir_path, pattern))


def find_unique_path_in_dir(
    dir_path: Path,
    pattern: str = ".*",
    ext: str | Sequence[str] = "",
    is_dir: bool = True,
    is_file: bool = True,
) -> Path | None:
    """
    Find a path in a given folder that fits the pattern and optionally the extensions.
    If the file found is not unique, an error is returned. If no file is found,
    None is returned. The flags 'is_dir' and 'is_file' determine the type of path
    we will be looking for. If both are True, then the path can be either.

    :param dir_path: The folder in which to search for the file.
    :param pattern: The pattern that we're trying to match.
    :param ext: string of the file extension to filter by or a sequence of such strings.
    :param is_dir: flag indicating whether we accept folder paths.
    :param is_file: flag indicating whether we accept file paths.

    :return: A Path object or None if no path is found.
    """

    match (is_dir, is_file):
        case (True, True):
            paths = get_paths_in_dir(dir_path, pattern, ext)
        case (True, False):
            if ext:
                raise ValueError("'ext' should not be set when looking only for dirs!")
            paths = get_dir_paths_in_dir(dir_path, pattern)
        case (False, True):
            paths = get_file_paths_in_dir(dir_path, pattern, ext)
        case _:
            raise ValueError("At least one of 'is_dir' and 'is_file' have to be True!")

    if (first_path := next(paths, None)) is not None:
        if next(paths, None) is not None:
            pattern = f"{pattern}.{ext}" if ext else pattern
            raise IOError(f"Found multiple paths with '{pattern}' in {dir_path}!")

    return first_path
