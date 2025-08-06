from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..arr import compress_de_axes, compress_de_axis, compress_sp_axes, compress_sp_axis
from .constants import DATA_DIR
from .database_dir import DatabaseDir
from .file_io import Data, Extension
from .utils import get_dir_paths_in_dir


class Database:

    database_dir_to_io_handler: Dict[str, Extension] = {
        "points": "npy",
        "masks": "npy",
        "params": "npy",
        "neigh": "npz",
        "n_counts": "pkl",
        "knn_dists": "pkl",
        "dists": "npz",
        "affs": "npz",
        "laps": "npz",
        "lap_eigvecs": "npy",
        "lap_eigvals": "npy",
        "wlpca_eigvals": "npy",
        "wlpca_eigvecs": "npy",
        "rmetric_eigvals": "npy",
        "rmetric_eigvecs": "npy",
        "eval_radii": "npy",
        "estimated_d": "npy",
        "ies": "json",
        "grads": "npy",
        "tslasso": "pkl",
    }

    @classmethod
    def _validate_dir_name(cls, dir_name: str) -> str:
        if dir_name in cls.database_dir_to_io_handler:
            return dir_name
        raise ValueError(f"Invalid dir type {dir_name}!")

    def __init__(
        self, 
        database_name: str, 
        dir_names: Optional[Sequence[str]] = None,
        mode='safe') -> None:

        self.path: Path = DATA_DIR / database_name

        if not self.path.exists():
            if mode == "safe":
                raise NotADirectoryError(f"Dir {self.path} does not exist!")
            else:
                if not self.path.exists():
                    print(f"Created dir '{self.path}'.")
                    self.path.mkdir()

        self.db_dirs: Dict[str, DatabaseDir] = {}

        dir_names = (
            dir_names
            if dir_names is not None
            else self.database_dir_to_io_handler.keys()
        )
        for dir_name in dir_names:
            self.open_dir(dir_name, mode=mode)

        for db_dir_path in get_dir_paths_in_dir(self.path):
            self._validate_dir_name(db_dir_path.name)

    def __getitem__(self, key: str) -> DatabaseDir | Data:

        key = key.split("|")
        if len(key) == 1:
            return self.db_dirs[key[0]]
        elif len(key) == 2:
            return self.db_dirs[key[0]][key[1]]
        elif len(key) == 3:
            return self._masked_load(*key)
        raise KeyError(f"Invalid key {key}!")

    def __setitem__(self, key: str, value: Data) -> None:
        key = key.split("|")
        if len(key) == 2:
            self.db_dirs[key[0]][key[1]] = value
        else:
            raise KeyError(f"Invalid item {key}!")

    def open_dir(self, dir_name: str, mode:str = "safe") -> DatabaseDir:
        io_handler = self.database_dir_to_io_handler[dir_name]
        db_dir = DatabaseDir(self.path / dir_name, io_handler, mode=mode)
        self.db_dirs[dir_name] = db_dir
        return db_dir

    def _masked_load(self, dir_name: str, data_name: str, mask_name: str) -> Data:

        data = self.db_dirs[dir_name][data_name]

        data_name = data_name.split("-")
        mask_name = mask_name.split("-")

        if all(mn == "all" for mn in mask_name):
            return data

        if len(data_name) != len(mask_name):
            raise ValueError(
                f"Data names' {data_name} length doesn't match mask names' length "
                f"{mask_name}!"
            )

        masks = tuple(
            self.db_dirs["masks"][f"{dn.split("_")[0]}_{mn}"]
            for (dn, mn) in zip(data_name, mask_name)
            if mn != "all"
        )

        if len(masks) == 0:
            raise ValueError("No mask was given!")
        elif len(masks) > 2:
            raise ValueError("At most 2 masks can be given!")

        match mask_name:

            case (_,) | ("all", _) | (_, "all"):
                axis = 0 if (len(masks) == 1 or masks[1] == "all") else 1
                if isinstance(data, np.ndarray):
                    return compress_de_axis(data, masks[0], axis=axis, in_place=True)
                return compress_sp_axis(
                    data, masks[0], axis=axis, reshape=True, in_place=True
                )
            case (_, _):
                if isinstance(data, np.ndarray):
                    return compress_de_axes(data, masks, in_place=True)
                return compress_sp_axes(data, masks, reshape=True, in_place=True)
            case _:
                raise ValueError(f"Invalid mask names {mask_name}!")
