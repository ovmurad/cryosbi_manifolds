import json
import pickle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import attr
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz as sp_load_npz
from scipy.sparse import save_npz as sp_save_npz

from .utils import find_unique_path_in_dir, get_path_ext

# Types

Extension: TypeAlias = Literal["pkl", "json", "npy", "npz"]
_io_extensions = set(Extension.__args__)

PathLike: TypeAlias = str | Path
Data = TypeVar("Data", Any, csr_matrix, NDArray)


def is_supported_io_ext(ext: str) -> TypeGuard[Extension]:
    return ext in _io_extensions


def _unlink_path(path: Path, **kwargs: Any) -> None:
    path.unlink(**kwargs)


def _validate_file_path(file_path: PathLike, ext: Extension) -> Path:

    file_path = Path(file_path)

    if file_path.is_dir():
        raise IsADirectoryError(f"'file_path' {file_path} is a directory!")

    if ext is not None:
        path_ext = get_path_ext(file_path)
        if path_ext != "" and path_ext != ext:
            raise IOError(f"Path ext of {file_path} is not {ext} as expected!")

    return file_path.with_suffix(f".{ext}")


class SaveFunc(Generic[Data]):

    def __init__(self, save_func: Callable[..., None], ext: Extension):
        self.save_func = save_func
        self.ext = ext

    def __call__(self, file_path: PathLike, data: Data, **kwargs: Any) -> None:
        file_path = _validate_file_path(file_path, self.ext)
        self.save_func(file_path, data, **kwargs)


class LoadFunc(Generic[Data]):

    def __init__(self, load_func: Callable[..., Data], ext: Extension):
        self.load_func = load_func
        self.ext = ext

    def __call__(self, file_path: PathLike, **kwargs: Any) -> Data:
        file_path = _validate_file_path(file_path, self.ext)
        return self.load_func(file_path, **kwargs)


class DeleteFunc:

    def __init__(self, delete_func: Callable[..., None], ext: Extension):
        self.delete_func = delete_func
        self.ext = ext

    def __call__(self, file_path: PathLike, **kwargs: Any) -> None:
        file_path = _validate_file_path(file_path, self.ext)
        self.delete_func(file_path, **kwargs)


# pickle load/save/delete


def _save_pkl(file_path: Path, data: Any, **kwargs: Any) -> None:
    """Save data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f, **kwargs)


def _load_pkl(file_path: Path, **kwargs: Any) -> Any:
    """Load data from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f, **kwargs)


save_pkl = SaveFunc[Any](_save_pkl, ext="pkl")
load_pkl = LoadFunc[Any](_load_pkl, ext="pkl")
delete_pkl = DeleteFunc(_unlink_path, ext="pkl")


# json load/save/delete


def _save_json(file_path: Path, data: Any, **kwargs: Any) -> None:
    """Write data to a file in JSON format."""
    with open(file_path, "w") as f:
        json.dump(data, f, **kwargs)


def _load_json(file_path: Path, **kwargs: Any) -> Any:
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f, **kwargs)


save_json = SaveFunc[Any](_save_json, ext="json")
load_json = LoadFunc[Any](_load_json, ext="json")
delete_json = DeleteFunc(_unlink_path, ext="json")


# npy load/save/delete

save_npy = SaveFunc[NDArray](np.save, ext="npy")
load_npy = LoadFunc[NDArray](np.load, ext="npy")
delete_npy = DeleteFunc(_unlink_path, ext="npy")

# npz load/save/delete

save_npz = SaveFunc[csr_matrix](sp_save_npz, ext="npz")
load_npz = LoadFunc[csr_matrix](sp_load_npz, ext="npz")
delete_npz = DeleteFunc(_unlink_path, ext="npz")


"""dispatches"""

load_func_dispatch: Dict[Extension, LoadFunc[Any]] = {
    "pkl": load_pkl,
    "json": load_json,
    "npy": load_npy,
    "npz": load_npz,
}


save_func_dispatch: Dict[Extension, SaveFunc[Any]] = {
    "pkl": save_pkl,
    "json": save_json,
    "npy": save_npy,
    "npz": save_npz,
}

delete_func_dispatch: Dict[Extension, DeleteFunc] = {
    "pkl": delete_pkl,
    "json": delete_json,
    "npy": delete_npy,
    "npz": delete_npz,
}

type_to_ext: Dict[type, Extension] = {
    np.ndarray: "npy",
    csr_matrix: "npz",
}


# any load/save/delete


def _maybe_infer_ext_from_path(file_path: Path, ext: Optional[Extension]) -> Extension:
    """Try to infer the extension either from the 'file_path' or check the dir
    in which the 'file_path' should be for the actual_path with the extension."""
    if ext is None and not (ext := get_path_ext(file_path)):

        file_path = find_unique_path_in_dir(
            dir_path=file_path.parent, pattern=f"{file_path.stem}.*", is_dir=False
        )
        if file_path is None:
            raise FileNotFoundError(f"No file {file_path} was found!")

        ext = get_path_ext(file_path)

        if not ext:
            raise IOError(f"Could not infer 'ext' for {file_path}!")

    if is_supported_io_ext(ext):
        return ext
    raise NotImplementedError(f"Extension '{ext}' is not supported!")


def _maybe_infer_ext_from_data_type(
    file_path: Path, data: Any, ext: Optional[Extension]
) -> Extension:
    """Try to infer the extension either from the 'file_path' or from the data type."""
    if ext is None:
        if not (ext := get_path_ext(file_path)):
            ext = type_to_ext.get(type(data), "pkl")

    if is_supported_io_ext(ext):
        return ext
    raise NotImplementedError(f"Extension '{ext}' is not supported!")


def load_any(file_path: Path, ext: Optional[Extension] = None, **kwargs: Any) -> Any:
    """
    Loads data from a file. The load function is chosen based on the given extension
    or inferred from the file extension of 'file_path' or from the file with the same
    name saved to disk.

    :param file_path: The path to the file that should be loaded.
    :param ext: The file extension. If no extension passed, 'ext' is inferred.
    :param kwargs: Additional keyword arguments to pass to the load function.

    :return: The data loaded from the file.
    """
    ext = _maybe_infer_ext_from_path(file_path, ext)
    return load_func_dispatch[ext](file_path, **kwargs)


def save_any(
    file_path: Path, data: Any, ext: Optional[Extension] = None, **kwargs: Any
) -> None:
    """
    Save data to file. The save function is chosen based on the given extension
    or inferred from the file extension of 'file_path' or from the type of the data.

    :param file_path: The path to the file where the data should be saved.
    :param data: The data to save.
    :param ext: The file extension. If no extension passed, 'ext' is inferred.
    :param kwargs: Additional keyword arguments to pass to the save function.
    """
    ext = _maybe_infer_ext_from_data_type(file_path, data, ext)
    return save_func_dispatch[ext](file_path, data, **kwargs)


def delete_any(file_path: Path, ext: Optional[Extension] = None, **kwargs: Any) -> None:
    """
    Delete data from file. The delete function is chosen based on the given extension
    or inferred from the file extension of 'file_path' or from the file with the same
    name saved to disk.

    :param file_path: The path to the file where the data should be saved.
    :param ext: The file extension. If no extension passed, 'ext' is inferred.
    :param kwargs: Additional keyword arguments to pass to the save function.
    """
    ext = _maybe_infer_ext_from_path(file_path, ext)
    return delete_func_dispatch[ext](file_path, **kwargs)


# IO Handlers


@attr.frozen(slots=True)
class IOHandler[Data]:
    """
    Helper class for handling the loading/saving/deleting of files belonging to one type
    of storage(npy, npz, etc.). One IOHandler for each type of data storage should
    be created and included in the factory below. All the instance variables are frozen,

    :ivar ext: extension to be used for the storage type.
    :ivar save_func: function to call for saving.
    :ivar load_func: function to call for loading.
    :ivar delete_func: function to call for deleting.
    """

    ext: Extension
    save_func: SaveFunc[Data]
    load_func: LoadFunc[Data]
    delete_func: DeleteFunc

    def save(self, file_path: PathLike, data: Data, **kwargs: Any) -> None:
        print(f"Saving {self.ext} file to {file_path}.")
        self.save_func(file_path, data, **kwargs)

    def load(self, file_path: PathLike, **kwargs: Any) -> Data:
        print(f"Loading {self.ext} file from {file_path}.")
        return self.load_func(file_path, **kwargs)

    def delete(self, file_path: PathLike, **kwargs: Any) -> None:
        print(f"Deleting {self.ext} file from {file_path}.")
        self.delete_func(file_path, **kwargs)


JsonIOHandler: IOHandler[Any] = IOHandler("json", save_json, load_json, delete_json)
PklIOHandler: IOHandler[Any] = IOHandler("pkl", save_pkl, load_pkl, delete_pkl)
NpyIOHandler: IOHandler[np.ndarray] = IOHandler("npy", save_npy, load_npy, delete_npy)
NpzIOHandler: IOHandler[csr_matrix] = IOHandler("npz", save_npz, load_npz, delete_npz)


def io_handler_factory(io_format: Extension) -> IOHandler:
    match io_format:
        case "pkl":
            return PklIOHandler
        case "json":
            return JsonIOHandler
        case "npy":
            return NpyIOHandler
        case "npz":
            return NpzIOHandler
        case _:
            raise NotImplementedError(f"IO format '{io_format}' is not implemented!")
